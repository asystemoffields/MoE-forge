"""gen-3 sparse-outlier quant: keep a small set of high-magnitude outliers at fp16 (value+index),
quantize the dense bulk to int4 grouped; search the outlier fraction by calib NLL self-validation."""

import io
import pickle
import zlib
import struct

import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM


def quantize_int4_grouped(arr_f32: np.ndarray, group_size: int = 128):
    n = arr_f32.size
    pad = (-n) % group_size
    if pad:
        arr_f32 = np.concatenate([arr_f32, np.zeros(pad, dtype=np.float32)])
    n_padded = arr_f32.size
    n_groups = n_padded // group_size
    groups = arr_f32.reshape(n_groups, group_size)
    max_abs = np.abs(groups).max(axis=1, keepdims=True).clip(min=1e-12)
    scales = (max_abs / 7.0).astype(np.float32)
    codes_f = np.clip(np.round(groups / scales), -7, 7).astype(np.int8)
    codes_u = (codes_f + 7).astype(np.uint8)
    codes_flat = codes_u.ravel()
    if codes_flat.size % 2:
        codes_flat = np.append(codes_flat, np.uint8(7))
    packed = (codes_flat[0::2] << 4) | (codes_flat[1::2] & 0x0F)
    return packed.astype(np.uint8), scales.ravel().astype(np.float16), n, pad


def dequantize_int4_grouped(packed: np.ndarray, scales_f16: np.ndarray,
                             orig_n: int, pad: int, group_size: int = 128):
    n_padded = orig_n + pad
    n_groups = n_padded // group_size
    hi = (packed >> 4) & 0x0F
    lo = packed & 0x0F
    codes_u = np.empty(hi.size + lo.size, dtype=np.uint8)
    codes_u[0::2] = hi
    codes_u[1::2] = lo
    codes_u = codes_u[:n_padded]
    codes_f = codes_u.astype(np.float32) - 7.0
    groups = codes_f.reshape(n_groups, group_size)
    scales = scales_f16.astype(np.float32).reshape(n_groups, 1)
    recon = (groups * scales).ravel()[:orig_n]
    return recon


def extract_outliers(arr_f32: np.ndarray, k: int):
    if k <= 0:
        return (np.array([], dtype=np.uint32), np.array([], dtype=np.float16), arr_f32.copy())
    flat = arr_f32.ravel()
    n = flat.size
    k = min(k, n)
    abs_flat = np.abs(flat)
    idx = np.argpartition(abs_flat, n - k)[n - k:]
    vals = flat[idx].astype(np.float16)
    residual = flat.copy()
    residual[idx] = 0.0
    return idx.astype(np.uint32), vals, residual.reshape(arr_f32.shape)


def restore_outliers(residual_flat: np.ndarray, idx: np.ndarray, vals_f16: np.ndarray):
    result = residual_flat.copy()
    if idx.size > 0:
        result[idx] = vals_f16.astype(np.float32)
    return result


def calib_nll(model, tokens):
    model.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for i in range(tokens.shape[0]):
            chunk = tokens[i:i+1]
            loss = model(input_ids=chunk, labels=chunk).loss
            total += float(loss)
            count += 1
    return total / max(count, 1)


def compress(model, calib_tokens, budget_bytes):
    GROUP_SIZE = 128
    OUTLIER_FRACS = [0.01, 0.02, 0.04]

    n_seqs = calib_tokens.shape[0]
    split = max(1, n_seqs // 2)
    val_half = calib_tokens[split:] if split < n_seqs else calib_tokens[:split]

    baseline_nll = calib_nll(model, val_half)

    config_dict = model.config.to_dict()
    state_dict = {k: v.detach().cpu().float().numpy() for k, v in model.state_dict().items()}

    best_result = None
    best_score = float('inf')

    for outlier_frac in OUTLIER_FRACS:
        quant = {}
        for key, arr in state_dict.items():
            if arr.ndim < 2:
                quant[key] = ("fp16", arr.astype(np.float16))
                continue
            flat = arr.ravel().astype(np.float32)
            k = max(0, int(flat.size * outlier_frac))
            out_idx, out_vals, residual = extract_outliers(arr, k)
            packed, scales, orig_n, pad = quantize_int4_grouped(residual.ravel(), GROUP_SIZE)
            quant[key] = ("soq", packed, scales, orig_n, pad, out_idx, out_vals, arr.shape)

        test_state = {}
        for key, record in quant.items():
            if record[0] == "fp16":
                test_state[key] = torch.from_numpy(record[1].astype(np.float32))
            else:
                _, packed, scales, orig_n, pad, out_idx, out_vals, shape = record
                recon_flat = dequantize_int4_grouped(packed, scales, orig_n, pad, GROUP_SIZE)
                recon_flat = restore_outliers(recon_flat, out_idx, out_vals)
                recon = recon_flat.reshape(shape).astype(np.float32)
                if not np.isfinite(recon).all():
                    recon = np.where(np.isfinite(recon), recon, 0.0)
                test_state[key] = torch.from_numpy(recon)

        cfg_copy = dict(config_dict)
        mt = cfg_copy.pop("model_type")
        cfg = AutoConfig.for_model(mt, **cfg_copy)
        test_model = AutoModelForCausalLM.from_config(cfg).eval()
        test_model.load_state_dict(test_state)
        nll_delta = calib_nll(test_model, val_half) - baseline_nll
        del test_model

        artifact_bytes = zlib.compress(pickle.dumps(
            {"config": config_dict, "quant": quant, "group_size": GROUP_SIZE}, protocol=4), level=1)
        sz = len(artifact_bytes)

        if nll_delta < 0.20:
            score = sz + max(0.0, nll_delta - 0.12) * 1e8
            if score < best_score:
                best_score = score
                best_result = (quant, artifact_bytes, nll_delta, sz)

    if best_result is None:
        outlier_frac = OUTLIER_FRACS[-1]
        quant = {}
        for key, arr in state_dict.items():
            if arr.ndim < 2:
                quant[key] = ("fp16", arr.astype(np.float16))
                continue
            flat = arr.ravel().astype(np.float32)
            k = max(0, int(flat.size * outlier_frac))
            out_idx, out_vals, residual = extract_outliers(arr, k)
            packed, scales, orig_n, pad = quantize_int4_grouped(residual.ravel(), GROUP_SIZE)
            quant[key] = ("soq", packed, scales, orig_n, pad, out_idx, out_vals, arr.shape)
        artifact_bytes = zlib.compress(pickle.dumps(
            {"config": config_dict, "quant": quant, "group_size": GROUP_SIZE}, protocol=4), level=6)
        best_result = (quant, artifact_bytes, 999.0, len(artifact_bytes))

    _, artifact_bytes, nll_delta, sz = best_result
    return artifact_bytes


def decompress(artifact: bytes):
    GROUP_SIZE_DEFAULT = 128
    blob = pickle.loads(zlib.decompress(artifact))
    config_dict = dict(blob["config"])
    quant = blob["quant"]
    group_size = blob.get("group_size", GROUP_SIZE_DEFAULT)

    model_type = config_dict.pop("model_type")
    cfg = AutoConfig.for_model(model_type, **config_dict)
    model = AutoModelForCausalLM.from_config(cfg)

    state = {}
    for key, record in quant.items():
        if record[0] == "fp16":
            arr = record[1].astype(np.float32)
        else:
            _, packed, scales, orig_n, pad, out_idx, out_vals, shape = record
            recon_flat = dequantize_int4_grouped(packed, scales, orig_n, pad, group_size)
            recon_flat = restore_outliers(recon_flat, out_idx, out_vals)
            arr = recon_flat.reshape(shape).astype(np.float32)
        if not np.isfinite(arr).all():
            arr = np.where(np.isfinite(arr), arr, 0.0)
        state[key] = torch.from_numpy(arr)

    model.load_state_dict(state)
    return model.eval()
