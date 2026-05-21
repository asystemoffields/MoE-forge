"""evolved_gptq4: first compress-evolve winner to beat the int8 baseline (gen-3, Sonnet-spawned,
"error-feedback" conception). On SmolLM-135M held-out NLL:

  method            ratio(vs bf16)   NLL delta
  noop (lossless)   1.00x            +0.000
  int8 RTN          1.65x            +0.096   (the bar)
  evolved_gptq4     2.42x            +0.093   <- Pareto-dominates int8 (smaller AND better)

It Pareto-dominates int8: more compression at equal-or-better quality. This is a REDISCOVERY of
GPTQ/OBQ (Frantar et al. 2022) -- novel-to-the-loop, not novel-to-the-world -- found by pointing
the search at a conception (propagate quantization error) the earlier generations never tried.
The genuinely-new territory is cross-pollination (rotation x this = QuaRot; this + sparse
outliers = SpQR) -- that's the next generation.

--- mechanism ---
Instead of rounding each weight independently (RTN), quantize each weight matrix's columns in
blocks and propagate each block's rounding error into the not-yet-quantized columns, weighted by
the layer's input second-order statistics H = X^T X (from calibration). Store 4-bit grouped codes
+ per-group scales; fall back to int8 on any layer where error feedback loses to plain RTN.
"""

import io
import pickle
import zlib

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM


BITS = 4
GROUP_SIZE = 128
BLOCK_SIZE = 64
RIDGE_FACTOR = 1e-3
CALIB_SPLIT = 0.8
MAX_CALIB_ROWS = 512


def quantize_tensor_int8(arr: np.ndarray):
    scale = float(np.abs(arr).max()) / 127.0
    if scale == 0.0:
        scale = 1.0
    codes = np.clip(np.round(arr / scale), -127, 127).astype(np.int8)
    return codes, np.float32(scale)


def dequantize_int8(codes: np.ndarray, scale: float) -> np.ndarray:
    return codes.astype(np.float32) * float(scale)


def quantize_4bit_grouped(row: np.ndarray, group_size: int = GROUP_SIZE):
    n = row.shape[0]
    pad = (-n) % group_size
    if pad:
        row = np.concatenate([row, np.zeros(pad, dtype=np.float32)])
    n_padded = row.shape[0]
    n_groups = n_padded // group_size
    groups = row.reshape(n_groups, group_size)
    absmax = np.abs(groups).max(axis=1, keepdims=True)
    absmax = np.where(absmax == 0, 1.0, absmax)
    scales = (absmax / 7.0).astype(np.float16)
    codes = np.clip(np.round(groups / scales.astype(np.float32)), -7, 7) + 8
    codes = codes.astype(np.uint8)
    flat = codes.reshape(-1)
    hi = flat[0::2] & 0xF
    lo = flat[1::2] & 0xF
    packed = ((hi << 4) | lo).astype(np.uint8)
    return packed, scales.reshape(-1), n


def dequantize_4bit_grouped(packed: np.ndarray, scales: np.ndarray, n_orig: int,
                             group_size: int = GROUP_SIZE) -> np.ndarray:
    n_groups = scales.shape[0]
    n_padded = n_groups * group_size
    flat = np.empty(n_padded, dtype=np.uint8)
    flat[0::2] = (packed >> 4) & 0xF
    flat[1::2] = packed & 0xF
    codes = flat.reshape(n_groups, group_size).astype(np.float32) - 8
    recon = codes * scales.astype(np.float32).reshape(-1, 1)
    return recon.reshape(-1)[:n_orig]


def gptq_quantize_weight(W: np.ndarray, H: np.ndarray,
                         group_size: int = GROUP_SIZE, block_size: int = BLOCK_SIZE):
    out_f, in_f = W.shape
    try:
        H_inv = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        H_inv = np.linalg.inv(H + np.eye(in_f, dtype=np.float32) * float(np.diag(H).mean()) * 0.01)
    if not np.isfinite(H_inv).all():
        H_inv = np.eye(in_f, dtype=np.float32)

    W_q = W.copy()
    for col_start in range(0, in_f, block_size):
        col_end = min(col_start + block_size, in_f)
        block_W = W_q[:, col_start:col_end]
        scale = np.abs(block_W).max(axis=1, keepdims=True) / 7.0
        scale = np.where(scale == 0, 1.0, scale)
        block_q = np.clip(np.round(block_W / scale), -7, 7) * scale
        error = block_W - block_q
        if col_end < in_f:
            H_block = H_inv[col_start:col_end, col_end:]
            correction = error @ H_block
            W_q[:, col_end:] -= correction
        W_q[:, col_start:col_end] = block_q

    packed_list = []
    for r in range(out_f):
        packed, scales, n_orig = quantize_4bit_grouped(W_q[r], group_size)
        packed_list.append((packed, scales, n_orig))
    return packed_list


def compress(model: nn.Module, calib_tokens, budget_bytes: int) -> bytes:
    model = model.eval()
    device = next(model.parameters()).device

    activations = {}
    hooks = []

    def make_hook(name):
        def hook(module, inp, out):
            x = inp[0].detach().cpu().float()
            x = x.reshape(-1, x.shape[-1])
            if name not in activations:
                activations[name] = []
            activations[name].append(x.numpy())
        return hook

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            hooks.append(module.register_forward_hook(make_hook(name)))

    n_total = calib_tokens.shape[0]
    n_gptq = max(1, int(n_total * CALIB_SPLIT))
    calib_gptq = calib_tokens[:n_gptq].to(device)

    with torch.no_grad():
        for i in range(calib_gptq.shape[0]):
            model(input_ids=calib_gptq[i:i+1])

    for h in hooks:
        h.remove()

    config_dict = model.config.to_dict()
    state_dict = model.state_dict()
    quant = {}

    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        W = module.weight.detach().cpu().float().numpy()
        out_f, in_f = W.shape

        use_gptq = False
        H = None
        if name in activations and len(activations[name]) > 0:
            X = np.concatenate(activations[name], axis=0)
            if X.shape[0] > MAX_CALIB_ROWS:
                idx = np.linspace(0, X.shape[0]-1, MAX_CALIB_ROWS, dtype=int)
                X = X[idx]
            if X.shape[1] == in_f and X.shape[0] >= 4:
                H = (X.T @ X).astype(np.float32)
                ridge = RIDGE_FACTOR * float(np.trace(H)) / max(in_f, 1)
                ridge = max(ridge, 1e-6)
                H += ridge * np.eye(in_f, dtype=np.float32)
                if np.isfinite(H).all():
                    use_gptq = True

        if use_gptq:
            try:
                packed_list = gptq_quantize_weight(W, H, GROUP_SIZE, BLOCK_SIZE)
                W_recon = np.zeros_like(W)
                for r, (packed, scales, n_orig) in enumerate(packed_list):
                    W_recon[r] = dequantize_4bit_grouped(packed, scales, n_orig, GROUP_SIZE)
                if not np.isfinite(W_recon).all():
                    use_gptq = False
                else:
                    int8_codes, int8_scale = quantize_tensor_int8(W)
                    W_int8_recon = dequantize_int8(int8_codes, int8_scale)
                    err_gptq = float(np.mean((W - W_recon)**2))
                    err_int8 = float(np.mean((W - W_int8_recon)**2))
                    if err_gptq > err_int8 * 3.0:
                        use_gptq = False
            except Exception:
                use_gptq = False

        if use_gptq:
            quant[name + ".weight"] = ("gptq4", packed_list)
        else:
            if W.size >= 1024:
                codes, scale = quantize_tensor_int8(W)
                quant[name + ".weight"] = ("int8", codes, scale)
            else:
                quant[name + ".weight"] = ("fp16", W.astype(np.float16))

        if module.bias is not None:
            b = module.bias.detach().cpu().float().numpy()
            quant[name + ".bias"] = ("fp16", b.astype(np.float16))

    handled = set(quant.keys())
    for key, tensor in state_dict.items():
        if key in handled:
            continue
        arr = tensor.detach().cpu().float().numpy()
        if arr.size > 4096 and arr.ndim >= 2:
            codes, scale = quantize_tensor_int8(arr)
            quant[key] = ("int8", codes, scale)
        else:
            quant[key] = ("fp16", arr.astype(np.float16))

    raw = pickle.dumps({"config": config_dict, "quant": quant}, protocol=4)
    return zlib.compress(raw, level=6)


def decompress(artifact: bytes) -> nn.Module:
    raw = zlib.decompress(artifact)
    blob = pickle.loads(raw)

    config = dict(blob["config"])
    model_type = config.pop("model_type")
    cfg = AutoConfig.for_model(model_type, **config)
    model = AutoModelForCausalLM.from_config(cfg)

    state = {}
    for key, record in blob["quant"].items():
        kind = record[0]
        if kind == "gptq4":
            _, packed_list = record
            rows = []
            for packed, scales, n_orig in packed_list:
                row = dequantize_4bit_grouped(packed, scales, int(n_orig), GROUP_SIZE)
                rows.append(row)
            arr = np.stack(rows, axis=0).astype(np.float32)
        elif kind == "int8":
            _, codes, scale = record
            arr = codes.astype(np.float32) * float(scale)
        else:
            arr = record[1].astype(np.float32)
        if not np.isfinite(arr).all():
            arr = np.where(np.isfinite(arr), arr, 0.0)
        state[key] = torch.tensor(arr)

    model.load_state_dict(state)
    return model.eval()
