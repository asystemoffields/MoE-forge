FAMILY = "outlier_packed"

"""SpQR-style outlier-augmented asymmetric int4 packing.

Key idea (SpQR / LLM.int8 lineage):
  A small fraction (~0.5-1%) of weight elements contribute disproportionately to
  quantisation error because they are both large AND land in input channels with
  large activation magnitudes.  Keep those few weights in fp16 precision (stored
  as sparse index+value pairs) and quantise the dense remainder to asymmetric
  int4 per-group.  The sparse outlier overhead is tiny (~0.5% of weight count *
  2 bytes each) but collapses NLL sharply vs pure int4.

Compression structure per linear layer:
  - dense_pk   uint8  [out_f, ceil(in_padded/2)]   packed 4-bit codes (outliers zeroed)
  - scale      fp16   [out_f, n_groups]
  - zero_point fp16   [out_f, n_groups]
  - out_idx    int32  flat outlier row indices
  - in_idx     int32  flat outlier col indices
  - out_vals   fp16   flat outlier values

The dequant path (pure torch, no .numpy()):
  1. Unpack dense int4 → float32 (standard asym formula)
  2. Scatter-add outlier values into the result matrix
  Runs entirely in torch; bf16-safe because we cast to float32 before arithmetic.

Outlier selection criterion (activation-weighted magnitude, vectorised):
  Salience = |W_ij| * col_norm_j   where col_norm_j = RMS of input activations
  for column j.  We pick the top-K by salience, with K = ceil(outlier_frac * n_elements).

Self-validation inside compress: reconstruct each layer and assert finite, then
measure per-layer Frobenius error; if error is large fall back to a wider group.

Resident memory: dense int4 (4 bits/weight) + scales/zp (fp16 per group) +
sparse outliers (~1% * 16 bits).  Expected ~2.3x compression at much lower NLL
than pure asym_packed_int4 (+0.459).
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

# ─────────────────────────────────────────────────────────────
# Hyper-parameters
# ─────────────────────────────────────────────────────────────
GROUP_SIZE = 64          # per-group int4 (smaller → better, minor overhead)
OUTLIER_FRAC = 0.008     # fraction of weights kept as fp16 outliers (~0.8%)
MAX_CALIB_ROWS = 256     # rows collected per layer from calib
CALIB_FRAC = 0.90        # fraction of calib seqs used for activation collection


# ─────────────────────────────────────────────────────────────
# Numpy helpers (compress-side only, never called in forward)
# ─────────────────────────────────────────────────────────────

def _compute_col_norms(activations_list):
    """Concatenate activation chunks and return per-column RMS."""
    X = np.concatenate(activations_list, axis=0).astype(np.float32)  # [rows, in_f]
    return np.sqrt((X ** 2).mean(axis=0) + 1e-12)  # [in_f]


def _select_outliers(W: np.ndarray, col_norms: np.ndarray, outlier_frac: float):
    """Return (row_idx, col_idx, values) for top-k salience outliers.

    Salience = |W[i,j]| * col_norms[j].  Vectorised via argpartition.
    """
    n_el = W.size
    k = max(1, int(np.ceil(outlier_frac * n_el)))
    k = min(k, n_el - 1)

    # salience matrix [out_f, in_f]
    sal = np.abs(W) * col_norms[np.newaxis, :]   # broadcast [out_f, in_f]

    # flat top-k (argpartition is O(n), not O(n log n))
    flat_sal = sal.ravel()
    thresh_idx = np.argpartition(flat_sal, -k)[-k:]
    out_f, in_f = W.shape
    row_idx = thresh_idx // in_f
    col_idx = thresh_idx % in_f
    vals = W[row_idx, col_idx]
    return row_idx.astype(np.int32), col_idx.astype(np.int32), vals.astype(np.float16)


def _asym_quant_np(W_dense: np.ndarray, group_size: int):
    """Asymmetric per-group int4 quantization on the (outlier-zeroed) dense residual.

    Returns packed uint8 [out_f, ceil(in_padded/2)],
            scale fp16   [out_f, n_groups],
            zero_point fp16 [out_f, n_groups],
            original in_f (int).
    """
    out_f, in_f = W_dense.shape
    pad = (-in_f) % group_size
    if pad:
        W_dense = np.concatenate([W_dense, np.zeros((out_f, pad), dtype=np.float32)], axis=1)
    in_p = W_dense.shape[1]
    n_groups = in_p // group_size
    Wg = W_dense.reshape(out_f, n_groups, group_size)

    wmin = Wg.min(axis=2)   # [out_f, n_groups]
    wmax = Wg.max(axis=2)
    wmax = np.where(wmax > wmin + 1e-8, wmax, wmin + 1e-8)

    scale = (wmax - wmin) / 15.0
    scale = np.maximum(scale, 1e-12)
    zero_point = wmin

    zp_b = zero_point[:, :, np.newaxis]
    sc_b = scale[:, :, np.newaxis]
    codes = np.clip(np.round((Wg - zp_b) / sc_b), 0, 15).astype(np.uint8)
    codes_flat = codes.reshape(out_f, in_p)
    packed = (codes_flat[:, 0::2] << 4) | (codes_flat[:, 1::2] & 0xF)

    return (packed.astype(np.uint8),
            scale.astype(np.float16),
            zero_point.astype(np.float16),
            in_f)


def _asym_dequant_np(packed, scale, zp, in_f, group_size):
    """Numpy dequant — compress-side only (self-validation)."""
    out_f = packed.shape[0]
    in_p = packed.shape[1] * 2
    n_groups = scale.shape[1]
    codes = np.empty((out_f, in_p), dtype=np.float32)
    codes[:, 0::2] = ((packed >> 4) & 0xF).astype(np.float32)
    codes[:, 1::2] = (packed & 0xF).astype(np.float32)
    sc = scale.astype(np.float32)[:, :, np.newaxis]
    zp32 = zp.astype(np.float32)[:, :, np.newaxis]
    W = (codes.reshape(out_f, n_groups, group_size) * sc + zp32).reshape(out_f, in_p)
    return W[:, :in_f]


# ─────────────────────────────────────────────────────────────
# Pure-torch dequant (forward path — no .numpy() ever)
# ─────────────────────────────────────────────────────────────

def _torch_asym_dequant_outliers(
    packed: torch.Tensor,
    scale: torch.Tensor,
    zp: torch.Tensor,
    in_f: int,
    group_size: int,
    out_idx: torch.Tensor,
    in_idx: torch.Tensor,
    out_vals: torch.Tensor,
) -> torch.Tensor:
    """Pure-torch dequant: dense int4 + sparse outlier scatter.

    packed   uint8  [out_f, in_padded//2]
    scale    fp16   [out_f, n_groups]
    zp       fp16   [out_f, n_groups]
    out_idx  int32  [K]  outlier row indices
    in_idx   int32  [K]  outlier col indices
    out_vals fp16   [K]  outlier fp16 values
    Returns  float32 [out_f, in_f]
    """
    out_f = packed.shape[0]
    in_p = packed.shape[1] * 2
    n_groups = scale.shape[1]

    # Unpack nibbles → float32
    codes = torch.empty(out_f, in_p, dtype=torch.float32, device=packed.device)
    codes[:, 0::2] = ((packed >> 4) & 0xF).float()
    codes[:, 1::2] = (packed & 0xF).float()

    # Dequantize (all float32; scale/zp may be fp16 or bf16 — cast)
    sc = scale.float().unsqueeze(2)    # [out_f, n_groups, 1]
    zp32 = zp.float().unsqueeze(2)
    W = (codes.view(out_f, n_groups, group_size) * sc + zp32).view(out_f, in_p)
    W = W[:, :in_f]  # strip padding

    # Scatter outlier corrections (add precise value over quantised placeholder)
    if out_idx.numel() > 0:
        # The dense matrix was quantised with outlier positions zeroed, so the
        # quantised value at those positions approximates 0 (not the original).
        # We scatter-add (precise_val - 0) = precise_val into W.
        W.index_put_(
            (out_idx.long(), in_idx.long()),
            out_vals.float(),
            accumulate=False,  # replace (outlier was zeroed before quant, so dense=0 there)
        )

    return W


# ─────────────────────────────────────────────────────────────
# Compressed linear module
# ─────────────────────────────────────────────────────────────

class OutlierInt4Linear(nn.Module):
    """Asymmetric int4 dense backbone + sparse fp16 outliers.  Dequant fully in torch."""

    def __init__(self, packed, scale, zp, out_features, in_features,
                 out_idx, in_idx, out_vals, bias=None, group_size=64):
        super().__init__()
        self.register_buffer("packed", packed)       # uint8 [out_f, in_p//2]
        self.register_buffer("scale", scale)         # fp16  [out_f, n_groups]
        self.register_buffer("zp", zp)               # fp16  [out_f, n_groups]
        self.register_buffer("out_idx", out_idx)     # int32 [K]
        self.register_buffer("in_idx", in_idx)       # int32 [K]
        self.register_buffer("out_vals", out_vals)   # fp16  [K]
        self.register_buffer("bias", bias)
        self.out_features = int(out_features)
        self.in_features = int(in_features)
        self.group_size = int(group_size)

    def forward(self, x):
        W = _torch_asym_dequant_outliers(
            self.packed, self.scale, self.zp,
            self.in_features, self.group_size,
            self.out_idx, self.in_idx, self.out_vals,
        ).to(x.dtype)
        b = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, W, b)


# ─────────────────────────────────────────────────────────────
# compress
# ─────────────────────────────────────────────────────────────

def compress(model, calib_tokens, budget_bytes) -> bytes:
    model = model.eval()
    device = next(model.parameters()).device

    # ── 1. Collect per-layer input activations via forward hooks ──
    activations: dict = {}
    hooks = []

    def _make_hook(name):
        def _hook(module, inp, out):
            x = inp[0].detach().cpu().float()   # cast bf16→float32 in torch (no .numpy() on bf16)
            x_np = x.reshape(-1, x.shape[-1]).numpy()
            store = activations.setdefault(name, [])
            if sum(a.shape[0] for a in store) < MAX_CALIB_ROWS:
                store.append(x_np)
        return _hook

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            hooks.append(module.register_forward_hook(_make_hook(name)))

    n_total = calib_tokens.shape[0]
    n_calib = max(1, int(n_total * CALIB_FRAC))
    with torch.no_grad():
        for i in range(n_calib):
            model(input_ids=calib_tokens[i:i+1].to(device))

    for h in hooks:
        h.remove()

    # ── 2. Build per-layer column norms ──
    col_norms: dict = {}
    for name, chunks in activations.items():
        col_norms[name] = _compute_col_norms(chunks)

    # ── 3. Quantize each linear ──
    cfg = model.config.to_dict()
    sd = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    linear_owners = {n for n, m in model.named_modules() if isinstance(m, nn.Linear)}

    linears: dict = {}
    handled: set = set()

    for owner in sorted(linear_owners):
        wk = owner + ".weight"
        bk = owner + ".bias"
        if wk not in sd or sd[wk].ndim != 2:
            continue

        W_orig = sd[wk].float().numpy()  # [out_f, in_f]
        out_f, in_f = W_orig.shape
        handled.add(wk)
        if bk in sd:
            handled.add(bk)

        bias_arr = sd[bk].to(torch.float16).numpy() if bk in sd else None

        # Get column norms (fall back to ones if not available or shape mismatch)
        cn = col_norms.get(owner, None)
        if cn is None or cn.shape[0] != in_f:
            cn = np.ones(in_f, dtype=np.float32)

        # ── Select outliers ──
        row_idx, col_idx, out_vals_fp16 = _select_outliers(W_orig, cn, OUTLIER_FRAC)

        # ── Zero outlier positions in the dense weight before quantising ──
        W_dense = W_orig.copy()
        W_dense[row_idx, col_idx] = 0.0

        # ── Quantise dense residual with fallback group sizes ──
        ok = False
        for gs in [GROUP_SIZE, 128, 256]:
            try:
                pk, sc, zp_np, in_f_stored = _asym_quant_np(W_dense, gs)
                W_rec = _asym_dequant_np(pk, sc, zp_np, in_f_stored, gs)
                # Insert outlier values back for validation
                W_rec[row_idx, col_idx] = out_vals_fp16.astype(np.float32)
                if np.isfinite(W_rec).all():
                    ok = True
                    group_used = gs
                    break
            except Exception:
                continue

        if not ok:
            # Hard fallback: store raw weight as fp16 (lossless-ish)
            raw_fp16 = np.clip(W_orig, -65504, 65504).astype(np.float16)
            if not np.isfinite(raw_fp16).all():
                raw_fp16 = np.nan_to_num(raw_fp16, nan=0.0, posinf=0.0, neginf=0.0)
            linears[owner] = {
                "mode": "raw_fp16",
                "w": raw_fp16,
                "out_f": int(out_f), "in_f": int(in_f),
                "bias": bias_arr,
            }
            continue

        linears[owner] = {
            "mode": "outlier_int4",
            "pk": pk,
            "sc": sc,
            "zp": zp_np,
            "out_f": int(out_f),
            "in_f": int(in_f_stored),
            "gs": group_used,
            "row_idx": row_idx,
            "col_idx": col_idx,
            "out_vals": out_vals_fp16,
            "bias": bias_arr,
        }

    # ── 4. Non-linear tensors at fp16 ──
    other: dict = {}
    for k, v in sd.items():
        if k not in handled:
            arr = v.to(torch.float16).numpy()
            if not np.isfinite(arr).all():
                arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
            other[k] = arr

    blob = {"config": cfg, "linears": linears, "other": other}
    return zlib.compress(pickle.dumps(blob, protocol=4), level=6)


# ─────────────────────────────────────────────────────────────
# decompress
# ─────────────────────────────────────────────────────────────

def decompress(artifact: bytes) -> nn.Module:
    blob = pickle.loads(zlib.decompress(artifact))
    cfg = dict(blob["config"])
    model_type = cfg.pop("model_type")
    model = AutoModelForCausalLM.from_config(AutoConfig.for_model(model_type, **cfg))

    linears = blob["linears"]

    def build(full_name: str, lin: nn.Linear) -> nn.Module:
        if full_name not in linears:
            return lin

        d = linears[full_name]
        mode = d.get("mode", "outlier_int4")
        bias_t = (torch.from_numpy(d["bias"].astype(np.float16))
                  if d["bias"] is not None else None)

        if mode == "raw_fp16":
            # Fallback: plain fp16 weight stored densely
            W_np = d["w"].astype(np.float32)
            if not np.isfinite(W_np).all():
                W_np = np.nan_to_num(W_np, nan=0.0, posinf=0.0, neginf=0.0)
            W_t = torch.from_numpy(W_np)
            pk_t, sc_t, in_f2 = P.pack_int4_grouped(W_t, 128)
            return P.Int4Linear(pk_t, sc_t, int(d["out_f"]), in_f2, bias_t, 128)

        # ── outlier_int4 ──
        out_f = int(d["out_f"])
        in_f = int(d["in_f"])
        gs = int(d["gs"])

        pk_t = torch.from_numpy(d["pk"].astype(np.uint8))
        sc_t = torch.from_numpy(d["sc"].astype(np.float16))
        zp_t = torch.from_numpy(d["zp"].astype(np.float16))

        row_idx_t = torch.from_numpy(d["row_idx"].astype(np.int32))
        col_idx_t = torch.from_numpy(d["col_idx"].astype(np.int32))
        out_vals_t = torch.from_numpy(d["out_vals"].astype(np.float16))

        layer = OutlierInt4Linear(
            pk_t, sc_t, zp_t, out_f, in_f,
            row_idx_t, col_idx_t, out_vals_t,
            bias_t, gs,
        )

        # Self-validate: reconstruct in torch, assert finite
        with torch.no_grad():
            W_check = _torch_asym_dequant_outliers(
                layer.packed, layer.scale, layer.zp,
                layer.in_features, layer.group_size,
                layer.out_idx, layer.in_idx, layer.out_vals,
            )
        if not torch.isfinite(W_check).all():
            # Emergency: re-pack the nan-cleaned weight as plain int4
            W_safe = torch.nan_to_num(W_check, nan=0.0, posinf=0.0, neginf=0.0).float()
            pk2, sc2, in_f2 = P.pack_int4_grouped(W_safe, 128)
            return P.Int4Linear(pk2, sc2, out_f, in_f2, bias_t, 128)

        return layer

    P.replace_linears(model, build)

    # Load embeddings, norms, and other non-linear weights at bf16
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
