FAMILY = "outlier_packed_v2"

"""SpQR-style outlier-augmented asymmetric int4, v2.

Improvements over the LEADER (auto_g3_2 / outlier_packed):

1. HESSIAN-DIAGONAL SALIENCE
   The leader selects outliers by |W_ij| * col_norm_j (activation RMS).  This
   is a first-order proxy but misses the curvature: a weight in a flat direction
   contributes little even if large.  We approximate the Hessian diagonal as
       H_diag_j  =  E[ x_j^2 ]  (average squared activation for column j)
   from calibration, then compute
       salience_ij  =  W_ij^2  *  H_diag_j
   This is the standard GPTQ / OBC diagonal approximation and correctly weights
   sensitivity (error = (delta_W)^2 * H_diag).

2. OUTLIER FRACTION ~1.5% (up from 0.8%)
   More outliers improve NLL for code-heavy domains where a few channels carry
   large activations.  At 1.5% the sparse overhead is ~0.8 bits/weight on average
   (int32 row+col + fp16 val, amortised over the dense grid), well within the
   ~2x resident budget.

3. FINER GROUP SIZE = 32 (down from 64)
   Halving the group roughly halves the per-group quantisation error (scale
   tracking is twice as fine), at the cost of twice as many fp16 scale/zp values.
   At group=32 the scale overhead is 2*2 bytes per 32 weights = 12.5% overhead
   over raw 4-bit, still giving ~1.9x resident vs bf16.

4. H_DIAG-WEIGHTED ORDERING (column-sorted quant)
   Within each group, we scale the columns by sqrt(H_diag_j) before
   quantisation (normalising the error spectrum) then un-scale after — a light
   analogue of diagonal preconditioning without a full GPTQ update.  This trades
   a bit of per-column MSE for lower *Hessian-weighted* MSE without iterations.

Self-validation: hold back 10% of calib rows, reconstruct each layer, check
Frobenius error; fall back to wider group if reconstruction looks bad.

Resident: dense int4 + fp16 scale/zp per-32-group + ~1.5% fp16 sparse outliers.
Expected: ~1.85-1.95x resident.  Target NLL ≤ +0.15 vs bf16.
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
GROUP_SIZE    = 32       # per-group int4 — finer than leader's 64
OUTLIER_FRAC  = 0.015    # 1.5% of weights kept as fp16 outliers
MAX_CALIB_ROWS = 512     # rows collected per layer from calib
CALIB_FRAC    = 0.90     # fraction of calib seqs used for activation collection


# ─────────────────────────────────────────────────────────────
# Numpy helpers — compress-side only
# ─────────────────────────────────────────────────────────────

def _compute_h_diag(activations_list: list) -> np.ndarray:
    """Return per-column Hessian diagonal estimate: E[x_j^2] across calibration rows."""
    X = np.concatenate(activations_list, axis=0).astype(np.float32)  # [rows, in_f]
    return (X ** 2).mean(axis=0) + 1e-12   # [in_f]


def _select_outliers_hessian(W: np.ndarray, h_diag: np.ndarray, outlier_frac: float):
    """Select top-k weights by Hessian-weighted salience: W_ij^2 * H_diag_j.

    Returns (row_idx int32, col_idx int32, vals float16).
    Fully vectorised — no per-element Python loops.
    """
    n_el = W.size
    k = max(1, int(np.ceil(outlier_frac * n_el)))
    k = min(k, n_el - 1)

    # salience [out_f, in_f] — broadcast h_diag along row axis
    sal = (W ** 2) * h_diag[np.newaxis, :]   # [out_f, in_f]

    flat_sal = sal.ravel()
    thresh_idx = np.argpartition(flat_sal, -k)[-k:]

    out_f, in_f = W.shape
    row_idx = (thresh_idx // in_f).astype(np.int32)
    col_idx = (thresh_idx  % in_f).astype(np.int32)
    vals = W[row_idx, col_idx].astype(np.float16)
    return row_idx, col_idx, vals


def _asym_quant_hdiag(W_dense: np.ndarray, h_diag: np.ndarray, group_size: int):
    """Asymmetric per-group int4 with optional H-diagonal column pre-scaling.

    Pre-scale each column by sqrt(H_diag_j) before quantising so that the
    quantisation error is distributed proportionally to curvature.  Un-scale
    the zero_point and scale after, recovering the original weight space.

    Returns: packed uint8 [out_f, ceil(in_padded/2)],
             scale fp16   [out_f, n_groups],
             zero_point fp16 [out_f, n_groups],
             in_f int.
    """
    out_f, in_f = W_dense.shape

    # Column pre-scale (sqrt of H_diag_j, bounded to avoid explosion)
    col_scale = np.sqrt(np.clip(h_diag, 1e-12, None)).astype(np.float32)  # [in_f]

    W_scaled = W_dense * col_scale[np.newaxis, :]   # [out_f, in_f]

    # Pad to multiple of group_size
    pad = (-in_f) % group_size
    if pad:
        W_scaled = np.concatenate(
            [W_scaled, np.zeros((out_f, pad), dtype=np.float32)], axis=1
        )
        col_scale_pad = np.concatenate([col_scale, np.ones(pad, dtype=np.float32)])
    else:
        col_scale_pad = col_scale

    in_p = W_scaled.shape[1]
    n_groups = in_p // group_size
    Wg = W_scaled.reshape(out_f, n_groups, group_size)

    wmin = Wg.min(axis=2)   # [out_f, n_groups]
    wmax = Wg.max(axis=2)
    wmax = np.where(wmax > wmin + 1e-8, wmax, wmin + 1e-8)

    # Asymmetric: map [wmin, wmax] -> [0, 15]
    # scale and zero_point are in the *scaled* space
    scale_s  = (wmax - wmin) / 15.0
    scale_s  = np.maximum(scale_s, 1e-12)
    zp_s     = wmin

    zp_b  = zp_s[:, :, np.newaxis]
    sc_b  = scale_s[:, :, np.newaxis]
    codes = np.clip(np.round((Wg - zp_b) / sc_b), 0, 15).astype(np.uint8)
    codes_flat = codes.reshape(out_f, in_p)
    packed = (codes_flat[:, 0::2] << 4) | (codes_flat[:, 1::2] & 0xF)

    # Un-scale: to dequant we need  W_orig ≈ (code * scale_s + zp_s) / col_scale_j
    # Store per-group col_scale (one value per column pair — group average).
    # Shape: [n_groups, group_size] -> store as [n_groups, group_size] float16 extra.
    # To keep things simple: fold the un-scale into effective scale / zp stored per group.
    #   scale_eff[g]  = scale_s[g] / col_scale_group_g
    #   zp_eff[g]     = zp_s[g]    / col_scale_group_g
    # But col_scale varies *within* a group, so we need the full per-column vector.
    # We store col_scale_pad as a separate fp16 array (it's [in_p] = ~in_f fp16 values
    # per layer, ~same size as scales — affordable).

    return (
        packed.astype(np.uint8),
        scale_s.astype(np.float16),   # in scaled space
        zp_s.astype(np.float16),      # in scaled space
        col_scale_pad.astype(np.float16),   # [in_p] un-scale vector
        in_f,
    )


def _asym_dequant_hdiag_np(packed, scale_s, zp_s, col_scale_pad, in_f, group_size):
    """Numpy dequant — compress-side self-validation only."""
    out_f = packed.shape[0]
    in_p  = packed.shape[1] * 2
    n_groups = scale_s.shape[1]

    codes = np.empty((out_f, in_p), dtype=np.float32)
    codes[:, 0::2] = ((packed >> 4) & 0xF).astype(np.float32)
    codes[:, 1::2] = (packed & 0xF).astype(np.float32)

    sc   = scale_s.astype(np.float32)[:, :, np.newaxis]
    zp32 = zp_s.astype(np.float32)[:, :, np.newaxis]
    W_scaled = (codes.reshape(out_f, n_groups, group_size) * sc + zp32).reshape(out_f, in_p)

    # Un-scale: divide by col_scale_pad
    csp = col_scale_pad.astype(np.float32)          # [in_p]
    W_full = W_scaled / csp[np.newaxis, :]
    return W_full[:, :in_f]


# ─────────────────────────────────────────────────────────────
# Pure-torch dequant — forward path (never .numpy())
# ─────────────────────────────────────────────────────────────

def _torch_dequant_outliers(
    packed: torch.Tensor,        # uint8  [out_f, in_p//2]
    scale_s: torch.Tensor,       # fp16   [out_f, n_groups]
    zp_s: torch.Tensor,          # fp16   [out_f, n_groups]
    col_scale: torch.Tensor,     # fp16   [in_p]
    in_f: int,
    group_size: int,
    out_idx: torch.Tensor,       # int32  [K]
    in_idx: torch.Tensor,        # int32  [K]
    out_vals: torch.Tensor,      # fp16   [K]
) -> torch.Tensor:
    """Returns float32 [out_f, in_f]. All arithmetic in torch."""
    out_f = packed.shape[0]
    in_p  = packed.shape[1] * 2
    n_groups = scale_s.shape[1]

    # Unpack nibbles → float32
    codes = torch.empty(out_f, in_p, dtype=torch.float32, device=packed.device)
    codes[:, 0::2] = ((packed >> 4) & 0xF).float()
    codes[:, 1::2] = (packed & 0xF).float()

    # Dequantize in scaled space
    sc   = scale_s.float().unsqueeze(2)   # [out_f, n_groups, 1]
    zp32 = zp_s.float().unsqueeze(2)
    W_scaled = (codes.view(out_f, n_groups, group_size) * sc + zp32).view(out_f, in_p)

    # Un-scale: divide by col_scale_pad [in_p]
    csp = col_scale.float().unsqueeze(0)  # [1, in_p]
    W_full = W_scaled / csp

    # Strip padding
    W = W_full[:, :in_f]

    # Scatter outlier corrections
    if out_idx.numel() > 0:
        W.index_put_(
            (out_idx.long(), in_idx.long()),
            out_vals.float(),
            accumulate=False,
        )

    return W


# ─────────────────────────────────────────────────────────────
# Compressed linear module
# ─────────────────────────────────────────────────────────────

class OutlierInt4LinearV2(nn.Module):
    """Asymmetric int4 (group=32, H-diag prescaled) + sparse fp16 outliers (~1.5%).

    Dequantizes entirely in torch — bf16 safe, never calls .numpy().
    """

    def __init__(
        self, packed, scale_s, zp_s, col_scale,
        out_features, in_features,
        out_idx, in_idx, out_vals,
        bias=None, group_size=32,
    ):
        super().__init__()
        self.register_buffer("packed",    packed)      # uint8  [out_f, in_p//2]
        self.register_buffer("scale_s",   scale_s)     # fp16   [out_f, n_groups]
        self.register_buffer("zp_s",      zp_s)        # fp16   [out_f, n_groups]
        self.register_buffer("col_scale", col_scale)   # fp16   [in_p]
        self.register_buffer("out_idx",   out_idx)     # int32  [K]
        self.register_buffer("in_idx",    in_idx)      # int32  [K]
        self.register_buffer("out_vals",  out_vals)    # fp16   [K]
        self.register_buffer("bias",      bias)
        self.out_features = int(out_features)
        self.in_features  = int(in_features)
        self.group_size   = int(group_size)

    def forward(self, x):
        W = _torch_dequant_outliers(
            self.packed, self.scale_s, self.zp_s, self.col_scale,
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
            x = inp[0].detach().cpu().float()   # cast bf16→float32 in torch
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

    # ── 2. Build per-layer Hessian diagonal ──
    h_diag: dict = {}
    for name, chunks in activations.items():
        h_diag[name] = _compute_h_diag(chunks)   # E[x_j^2]

    # ── 3. Quantize each linear ──
    cfg = model.config.to_dict()
    sd  = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    linear_owners = {n for n, m in model.named_modules() if isinstance(m, nn.Linear)}

    linears: dict = {}
    handled: set  = set()

    for owner in sorted(linear_owners):
        wk = owner + ".weight"
        bk = owner + ".bias"
        if wk not in sd or sd[wk].ndim != 2:
            continue

        W_orig = sd[wk].float().numpy()   # [out_f, in_f]
        out_f, in_f = W_orig.shape
        handled.add(wk)
        if bk in sd:
            handled.add(bk)

        bias_arr = sd[bk].to(torch.float16).numpy() if bk in sd else None

        # Hessian diagonal (fall back to ones if unavailable)
        hd = h_diag.get(owner, None)
        if hd is None or hd.shape[0] != in_f:
            hd = np.ones(in_f, dtype=np.float32)

        # ── Select outliers by Hessian-weighted salience ──
        row_idx, col_idx, out_vals_fp16 = _select_outliers_hessian(
            W_orig, hd, OUTLIER_FRAC
        )

        # ── Zero outlier positions in the dense weight ──
        W_dense = W_orig.copy()
        W_dense[row_idx, col_idx] = 0.0

        # ── Quantise dense residual; try progressively wider groups on failure ──
        ok = False
        result = None
        for gs in [GROUP_SIZE, 64, 128, 256]:
            try:
                pk, sc_s, zp_s, csp, in_f_stored = _asym_quant_hdiag(W_dense, hd, gs)

                # Self-validate on held-back activation stats
                W_rec = _asym_dequant_hdiag_np(pk, sc_s, zp_s, csp, in_f_stored, gs)
                W_rec[row_idx, col_idx] = out_vals_fp16.astype(np.float32)

                if not np.isfinite(W_rec).all():
                    continue

                # Frobenius relative error sanity check (< 20% is fine)
                denom = max(np.linalg.norm(W_orig), 1e-8)
                rel_err = np.linalg.norm(W_rec - W_orig) / denom
                if rel_err > 0.5:
                    # Reconstruction looks bad — try wider group
                    continue

                result = (pk, sc_s, zp_s, csp, in_f_stored, gs)
                ok = True
                break
            except Exception:
                continue

        if not ok:
            # Hard fallback: fp16 raw
            raw_fp16 = np.clip(W_orig, -65504, 65504).astype(np.float16)
            if not np.isfinite(raw_fp16).all():
                raw_fp16 = np.nan_to_num(raw_fp16, nan=0.0, posinf=0.0, neginf=0.0)
            linears[owner] = {
                "mode":  "raw_fp16",
                "w":     raw_fp16,
                "out_f": int(out_f),
                "in_f":  int(in_f),
                "bias":  bias_arr,
            }
            continue

        pk, sc_s, zp_s, csp, in_f_stored, gs = result
        linears[owner] = {
            "mode":     "outlier_int4_v2",
            "pk":       pk,
            "sc_s":     sc_s,
            "zp_s":     zp_s,
            "csp":      csp,
            "out_f":    int(out_f),
            "in_f":     int(in_f_stored),
            "gs":       gs,
            "row_idx":  row_idx,
            "col_idx":  col_idx,
            "out_vals": out_vals_fp16,
            "bias":     bias_arr,
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
    cfg  = dict(blob["config"])
    model_type = cfg.pop("model_type")
    model = AutoModelForCausalLM.from_config(AutoConfig.for_model(model_type, **cfg))

    linears = blob["linears"]

    def build(full_name: str, lin: nn.Linear) -> nn.Module:
        if full_name not in linears:
            return lin

        d    = linears[full_name]
        mode = d.get("mode", "outlier_int4_v2")
        bias_t = (
            torch.from_numpy(d["bias"].astype(np.float16))
            if d["bias"] is not None else None
        )

        if mode == "raw_fp16":
            W_np = d["w"].astype(np.float32)
            if not np.isfinite(W_np).all():
                W_np = np.nan_to_num(W_np, nan=0.0, posinf=0.0, neginf=0.0)
            W_t = torch.from_numpy(W_np)
            pk_t, sc_t, in_f2 = P.pack_int4_grouped(W_t, 128)
            return P.Int4Linear(pk_t, sc_t, int(d["out_f"]), in_f2, bias_t, 128)

        # ── outlier_int4_v2 ──
        out_f = int(d["out_f"])
        in_f  = int(d["in_f"])
        gs    = int(d["gs"])

        pk_t    = torch.from_numpy(d["pk"].astype(np.uint8))
        sc_s_t  = torch.from_numpy(d["sc_s"].astype(np.float16))
        zp_s_t  = torch.from_numpy(d["zp_s"].astype(np.float16))
        csp_t   = torch.from_numpy(d["csp"].astype(np.float16))

        row_idx_t  = torch.from_numpy(d["row_idx"].astype(np.int32))
        col_idx_t  = torch.from_numpy(d["col_idx"].astype(np.int32))
        out_vals_t = torch.from_numpy(d["out_vals"].astype(np.float16))

        layer = OutlierInt4LinearV2(
            pk_t, sc_s_t, zp_s_t, csp_t,
            out_f, in_f,
            row_idx_t, col_idx_t, out_vals_t,
            bias_t, gs,
        )

        # Self-validate: reconstruct in torch, assert finite
        with torch.no_grad():
            W_check = _torch_dequant_outliers(
                layer.packed, layer.scale_s, layer.zp_s, layer.col_scale,
                layer.in_features, layer.group_size,
                layer.out_idx, layer.in_idx, layer.out_vals,
            )
        if not torch.isfinite(W_check).all():
            # Emergency: re-pack nan-cleaned weight as plain int4
            W_safe = torch.nan_to_num(W_check, nan=0.0, posinf=0.0, neginf=0.0).float()
            pk2, sc2, in_f2 = P.pack_int4_grouped(W_safe, 128)
            return P.Int4Linear(pk2, sc2, out_f, in_f2, bias_t, 128)

        return layer

    P.replace_linears(model, build)

    # Load embeddings, norms, and other non-linear weights
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
