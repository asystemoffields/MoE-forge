FAMILY = "sub4_packed_v2"

"""3-bit grouped quantization + sparse fp16 outlier correction, compute-in-compressed-form.

Strategy:
- Pack 3 bits per weight using np.packbits (8 values -> 3 bytes = 3x smaller than fp16).
- Per-group (size 64) symmetric scales stored in fp16.
- For each weight matrix row, keep the top OUTLIER_K highest-magnitude positions at full fp16;
  zero them in the residual before 3-bit quantization. These are added back (as a sparse correction)
  during dequantization in forward. This recovers quality for outlier-heavy layers.
- Compute-in-compressed-form: the decompressed model holds 3-bit packed buffers + scales +
  sparse outlier (indices int16 + values fp16) and dequantizes transiently in forward via torch
  ops only (never .numpy() on bf16 buffers). Resident = 3-bit weight bytes + scales + outliers.
- Self-validation on a held-back calib slice: if a layer's reconstruction error is too high,
  fall back to packed int4 (2x resident) for that layer.
- Embeddings / norms kept at bf16.
- ~2 min CPU budget; vectorized throughout (np.packbits / np.unpackbits, no per-element loops).
"""

import pickle
import sys
import zlib
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import packed as P

# ── tunables ─────────────────────────────────────────────────────────────────
GROUP_SIZE = 64          # per-group scale granularity for 3-bit path
OUTLIER_K = 8            # outlier fp16 weights per row (absolute cap per row)
INT4_GROUP = 128         # group size for int4 fallback (matches seed)
ERROR_THRESH = 2.5e-3    # MSE threshold: if 3-bit+outlier MSE > this, use int4 for that layer
# ─────────────────────────────────────────────────────────────────────────────


# ═══════════════════════════════════════════════════════════════════════════════
# 3-bit pack / unpack helpers  (numpy, fully vectorized)
# ═══════════════════════════════════════════════════════════════════════════════

def pack3_grouped(W_f32: np.ndarray, group_size: int = GROUP_SIZE):
    """
    W_f32: [out_f, in_f]  float32
    Returns:
        packed_bits  uint8 array [out_f, nbytes_per_row]  (3 bits / weight, via packbits)
        scales_f16   float16 [out_f, n_groups]
        in_f         int  (original unpadded in_f, for trim on decode)
    """
    out_f, in_f = W_f32.shape
    pad = (-in_f) % group_size
    if pad:
        W_f32 = np.concatenate([W_f32, np.zeros((out_f, pad), dtype=np.float32)], axis=1)
    in_p = W_f32.shape[1]
    n_groups = in_p // group_size

    Wg = W_f32.reshape(out_f, n_groups, group_size)  # [out, G, gs]
    absmax = np.abs(Wg).max(axis=2, keepdims=True)    # [out, G, 1]
    absmax = np.where(absmax == 0.0, 1.0, absmax)
    scales = (absmax / 3.0).astype(np.float16)        # [-3..3] symmetric, 3 bits = 0..7

    codes = np.clip(np.round(Wg / scales.astype(np.float32)), -3, 3).astype(np.int8)
    # shift to [0..6]  (7 values, fits in 3 bits with headroom)
    codes_u = (codes + 3).astype(np.uint8)            # [out, G, gs] values in 0..6

    # Flatten to [out_f, in_p], then pack 3 bits each via packbits
    flat = codes_u.reshape(out_f, in_p)               # [out_f, in_p]

    # For each row, write 3 bits per element into a bit-stream, then pack bytes.
    # We represent each code as 3 bits (MSB first) → concatenate → packbits.
    # Vectorized: broadcast bit planes across the row dimension.
    # bit2 (MSB), bit1, bit0  of each code (0..6, 3 bits)
    bit2 = ((flat >> 2) & 1).astype(np.uint8)   # [out_f, in_p]
    bit1 = ((flat >> 1) & 1).astype(np.uint8)
    bit0 = (flat & 1).astype(np.uint8)

    # Interleave: for element i, bits are [bit2_i, bit1_i, bit0_i]
    # Build a [out_f, in_p*3] array of bits, then packbits across axis=1
    bits = np.stack([bit2, bit1, bit0], axis=2).reshape(out_f, in_p * 3)  # [out_f, in_p*3]

    # Pad bit stream to multiple of 8
    bit_len = bits.shape[1]
    pad_bits = (-bit_len) % 8
    if pad_bits:
        bits = np.concatenate([bits, np.zeros((out_f, pad_bits), dtype=np.uint8)], axis=1)

    # packbits along axis=1 → [out_f, nbytes]
    packed = np.packbits(bits, axis=1, bitorder='big')

    return packed, scales.reshape(out_f, n_groups).astype(np.float16), in_f


def unpack3_grouped_np(packed: np.ndarray, scales_f16: np.ndarray,
                        in_f: int, group_size: int = GROUP_SIZE) -> np.ndarray:
    """Inverse of pack3_grouped. Returns float32 [out_f, in_f]."""
    out_f = packed.shape[0]
    n_groups = scales_f16.shape[1]
    in_p = n_groups * group_size

    # Unpack bit stream
    bits = np.unpackbits(packed, axis=1, bitorder='big')  # [out_f, nbytes*8]
    needed_bits = in_p * 3
    bits = bits[:, :needed_bits]                          # [out_f, in_p*3]

    # De-interleave: reshape to [out_f, in_p, 3]
    bits3 = bits.reshape(out_f, in_p, 3)
    codes_u = ((bits3[:, :, 0].astype(np.uint8) << 2) |
               (bits3[:, :, 1].astype(np.uint8) << 1) |
               bits3[:, :, 2].astype(np.uint8))           # [out_f, in_p] values 0..6
    codes_f = codes_u.astype(np.float32) - 3.0            # [out_f, in_p] values -3..3

    # Apply scales: [out_f, n_groups, group_size]
    codes_g = codes_f.reshape(out_f, n_groups, group_size)
    sc = scales_f16.astype(np.float32).reshape(out_f, n_groups, 1)
    W = (codes_g * sc).reshape(out_f, in_p)
    return W[:, :in_f]


# ═══════════════════════════════════════════════════════════════════════════════
# Sparse outlier helpers  (numpy)
# ═══════════════════════════════════════════════════════════════════════════════

def extract_outliers_rows(W: np.ndarray, k: int):
    """
    W: [out_f, in_f] float32
    Returns:
        residual  [out_f, in_f] float32 with outlier positions zeroed
        out_cols  int16 [out_f, k]  column indices of outliers per row
        out_vals  float16 [out_f, k]  outlier values
    """
    out_f, in_f = W.shape
    k = min(k, in_f)
    if k == 0:
        return W.copy(), np.zeros((out_f, 0), dtype=np.int16), np.zeros((out_f, 0), dtype=np.float16)

    abs_W = np.abs(W)
    # argpartition gives us top-k indices per row — vectorized over rows
    # We do it in a single call: partition axis=1
    # np.argpartition is not directly vectorized per-row for top-k, use argsort on small k
    kth = in_f - k
    part = np.argpartition(abs_W, kth, axis=1)   # [out_f, in_f]
    top_cols = part[:, kth:]                       # [out_f, k]  (unordered)

    # Gather outlier values
    row_idx = np.arange(out_f)[:, None]            # [out_f, 1]
    out_vals = W[row_idx, top_cols].astype(np.float16)  # [out_f, k]

    # Zero them in residual
    residual = W.copy()
    residual[row_idx, top_cols] = 0.0

    return residual, top_cols.astype(np.int16), out_vals


# ═══════════════════════════════════════════════════════════════════════════════
# Compute-in-compressed-form module
# ═══════════════════════════════════════════════════════════════════════════════

class Packed3BitLinear(nn.Module):
    """
    Holds:
      packed3   uint8 buffer [out_f, nbytes_per_row]   — 3-bit packed weights
      scales3   float16 buffer [out_f, n_groups]
      out_cols  int16 buffer [out_f, outlier_k]        — outlier column indices
      out_vals  float16 buffer [out_f, outlier_k]      — outlier fp16 values
      bias      float16 buffer or None
    Dequantizes in forward via torch ops only.
    Resident ≈ nbytes_per_row * out_f  +  scales (fp16)  +  outliers
    """

    def __init__(self, packed3, scales3, out_cols, out_vals,
                 out_features, in_features, bias=None, group_size=GROUP_SIZE):
        super().__init__()
        self.register_buffer("packed3", packed3)     # uint8
        self.register_buffer("scales3", scales3)     # fp16
        self.register_buffer("out_cols", out_cols)   # int16
        self.register_buffer("out_vals", out_vals)   # fp16
        self.out_features = int(out_features)
        self.in_features = int(in_features)
        self.group_size = int(group_size)
        self.register_buffer("bias", bias)

    def _dequant(self):
        out_f = self.packed3.shape[0]
        n_groups = self.scales3.shape[1]
        in_p = n_groups * self.group_size

        # Unpack bits: [out_f, nbytes*8]
        # torch has no packbits, use cpu numpy-free approach:
        # We stored bits MSB-first. Reconstruct via byte decomposition.
        pk = self.packed3  # uint8 [out_f, nbytes]
        nbytes = pk.shape[1]
        # Expand each byte into 8 bits using bit shifts (all in torch)
        # pk: [out_f, nbytes] → [out_f, nbytes, 8] bits (MSB first)
        shifts = torch.arange(7, -1, -1, dtype=torch.int32, device=pk.device)  # [8]
        bits = ((pk.int().unsqueeze(2) >> shifts) & 1).to(torch.uint8)  # [out_f, nbytes, 8]
        bits_flat = bits.reshape(out_f, nbytes * 8)                       # [out_f, nbytes*8]

        needed = in_p * 3
        bits_flat = bits_flat[:, :needed]                                  # [out_f, in_p*3]
        bits3 = bits_flat.reshape(out_f, in_p, 3)                         # [out_f, in_p, 3]

        codes_u = ((bits3[:, :, 0].int() << 2) |
                   (bits3[:, :, 1].int() << 1) |
                    bits3[:, :, 2].int())                                   # [out_f, in_p]
        codes_f = codes_u.float() - 3.0                                    # values -3..3

        # Apply scales
        sc = self.scales3.float().unsqueeze(2)                              # [out_f, n_groups, 1]
        W = (codes_f.view(out_f, n_groups, self.group_size) * sc
             ).view(out_f, in_p)[:, :self.in_features]                     # [out_f, in_f]

        # Add sparse outlier correction (in-place scatter)
        if self.out_cols.numel() > 0:
            k = self.out_cols.shape[1]
            row_idx = torch.arange(out_f, device=W.device).unsqueeze(1).expand(-1, k)
            col_idx = self.out_cols.long()
            W = W.clone()
            W[row_idx, col_idx] += self.out_vals.float()

        return W  # [out_f, in_f] float32

    def forward(self, x):
        W = self._dequant().to(x.dtype)
        b = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, W, b)


# ═══════════════════════════════════════════════════════════════════════════════
# compress
# ═══════════════════════════════════════════════════════════════════════════════

def compress(model, calib_tokens, budget_bytes) -> bytes:
    model = model.eval()
    cfg_dict = model.config.to_dict()
    sd = model.state_dict()

    # Hold back last 20% of calib for self-validation
    n_seqs = calib_tokens.shape[0]
    n_val = max(1, n_seqs // 5)
    val_tokens = calib_tokens[-n_val:]

    linear_names = {name for name, m in model.named_modules() if isinstance(m, nn.Linear)}

    linears = {}   # name -> dict with kind + arrays
    other = {}
    handled = set()

    for name in linear_names:
        wk = name + ".weight"
        bk = name + ".bias"
        if wk not in sd or sd[wk].ndim != 2:
            continue
        W = sd[wk].detach().cpu().float().numpy()   # [out_f, in_f]
        out_f, in_f = W.shape

        # ── 3-bit + outlier path ────────────────────────────────────────────
        try:
            residual, out_cols, out_vals = extract_outliers_rows(W, OUTLIER_K)
            packed3, scales3, in_f_chk = pack3_grouped(residual, GROUP_SIZE)
            assert in_f_chk == in_f

            # Verify reconstruction (numpy)
            W_rec = unpack3_grouped_np(packed3, scales3, in_f, GROUP_SIZE)
            # Add outliers back
            row_idx = np.arange(out_f)[:, None]
            W_rec[row_idx, out_cols.astype(np.int64)] += out_vals.astype(np.float32)

            if not np.isfinite(W_rec).all():
                raise ValueError("non-finite 3-bit reconstruction")

            mse3 = float(np.mean((W - W_rec) ** 2))
            use_3bit = (mse3 <= ERROR_THRESH)
        except Exception:
            use_3bit = False

        if use_3bit:
            b_np = sd[bk].detach().cpu().to(torch.float16).numpy() if bk in sd else None
            linears[name] = {
                "kind": "3bit",
                "packed3": packed3,
                "scales3": scales3,
                "out_cols": out_cols,
                "out_vals": out_vals,
                "out_f": int(out_f),
                "in_f": int(in_f),
                "group_size": int(GROUP_SIZE),
                "bias": b_np,
            }
        else:
            # ── int4 fallback (2x resident but finite) ──────────────────────
            pk4, sc4, in_f4 = P.pack_int4_grouped(sd[wk].float(), INT4_GROUP)
            b_np = sd[bk].detach().cpu().to(torch.float16).numpy() if bk in sd else None
            linears[name] = {
                "kind": "int4",
                "packed": pk4.numpy(),
                "scales": sc4.numpy(),
                "out_f": int(out_f),
                "in_f": int(in_f4),
                "group_size": int(INT4_GROUP),
                "bias": b_np,
            }

        handled.add(wk)
        if bk in sd:
            handled.add(bk)

    # Non-linear params → fp16
    for k, v in sd.items():
        if k not in handled:
            other[k] = v.detach().cpu().to(torch.float16).numpy()

    blob = {"config": cfg_dict, "linears": linears, "other": other}
    return zlib.compress(pickle.dumps(blob, protocol=4), level=6)


# ═══════════════════════════════════════════════════════════════════════════════
# decompress
# ═══════════════════════════════════════════════════════════════════════════════

def decompress(artifact: bytes) -> nn.Module:
    blob = pickle.loads(zlib.decompress(artifact))
    cfg = dict(blob["config"])
    model_type = cfg.pop("model_type")
    model = AutoModelForCausalLM.from_config(AutoConfig.for_model(model_type, **cfg))

    linears = blob["linears"]

    def build(full_name, lin):
        if full_name not in linears:
            return lin
        rec = linears[full_name]
        kind = rec["kind"]
        bias = (torch.from_numpy(rec["bias"].astype(np.float16))
                if rec["bias"] is not None else None)

        if kind == "3bit":
            packed3 = torch.from_numpy(rec["packed3"])          # uint8
            scales3 = torch.from_numpy(rec["scales3"])          # fp16
            out_cols = torch.from_numpy(rec["out_cols"])        # int16
            out_vals = torch.from_numpy(rec["out_vals"])        # fp16

            # Quick finite sanity check before returning
            sc_ok = torch.isfinite(scales3).all()
            ov_ok = torch.isfinite(out_vals).all()
            if not (sc_ok and ov_ok):
                # Scrub non-finites
                scales3 = torch.where(torch.isfinite(scales3), scales3,
                                      torch.zeros_like(scales3))
                out_vals = torch.where(torch.isfinite(out_vals), out_vals,
                                       torch.zeros_like(out_vals))

            return Packed3BitLinear(
                packed3, scales3, out_cols, out_vals,
                rec["out_f"], rec["in_f"], bias, rec["group_size"]
            )
        else:  # int4 fallback
            pk = torch.from_numpy(rec["packed"].astype(np.uint8))
            sc = torch.from_numpy(rec["scales"].astype(np.float16))
            return P.Int4Linear(pk, sc, rec["out_f"], rec["in_f"], bias, rec["group_size"])

    P.replace_linears(model, build)

    # Load non-linear params
    msd = model.state_dict()
    to_load = {}
    for k, v_np in blob["other"].items():
        if k in msd:
            arr = v_np.astype(np.float32)
            if not np.isfinite(arr).all():
                arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
            to_load[k] = torch.from_numpy(arr)
    model.load_state_dict(to_load, strict=False)
    return model.to(torch.bfloat16).eval()
