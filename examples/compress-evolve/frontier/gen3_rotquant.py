"""gen-3 rotate-then-quantize: seeded sign+permutation incoherence transform per matrix, then
4-bit per-group quant in the transformed space; store seed (not the matrix); fall back to int8
if reconstruction error is high. (Incoherence-processing / QuaRot-flavored.)"""

import io
import pickle
import struct
import zlib

import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM


def _make_rotation(n, seed):
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=n).astype(np.float32)
    perm = rng.permutation(n).astype(np.int32)
    return signs, perm


def _apply_rotation(W, signs, perm):
    return W[:, perm] * signs[np.newaxis, :]


def _invert_rotation(W_rot, signs, perm):
    inv_perm = np.empty_like(perm)
    inv_perm[perm] = np.arange(len(perm), dtype=np.int32)
    return (W_rot[:, inv_perm] / signs[np.newaxis, inv_perm])


def _quantize_int4_pergroup(arr_2d, group_size=128):
    rows, cols = arr_2d.shape
    pad = (group_size - cols % group_size) % group_size
    if pad:
        arr_2d = np.concatenate([arr_2d, np.zeros((rows, pad), dtype=np.float32)], axis=1)
    cols_pad = arr_2d.shape[1]
    n_groups = cols_pad // group_size
    flat = arr_2d.reshape(rows * n_groups, group_size)
    amax = np.abs(flat).max(axis=1, keepdims=True)
    amax = np.where(amax == 0, 1.0, amax)
    scales = (amax / 7.0).astype(np.float16)
    scales_f32 = scales.astype(np.float32)
    codes = np.clip(np.round(flat / scales_f32), -7, 7).astype(np.int8)
    return codes, scales, (rows, cols), pad, group_size


def _dequantize_int4_pergroup(codes, scales, orig_shape, pad, group_size):
    rows, cols = orig_shape
    cols_pad = cols + pad
    n_groups = cols_pad // group_size
    flat_codes = codes.reshape(rows * n_groups, group_size).astype(np.float32)
    flat_scales = scales.reshape(rows * n_groups, 1).astype(np.float32)
    flat_recon = flat_codes * flat_scales
    recon = flat_recon.reshape(rows, cols_pad)
    if pad:
        recon = recon[:, :cols]
    return recon


def _pack_int4(codes_int8):
    u = (codes_int8.ravel() + 7).astype(np.uint8)
    if len(u) % 2 != 0:
        u = np.append(u, np.uint8(0))
    packed = (u[0::2] & 0x0F) | ((u[1::2] & 0x0F) << 4)
    return packed.astype(np.uint8)


def _unpack_int4(packed, n_elements):
    lo = (packed & 0x0F).astype(np.int8)
    hi = ((packed >> 4) & 0x0F).astype(np.int8)
    interleaved = np.empty(2 * len(packed), dtype=np.int8)
    interleaved[0::2] = lo
    interleaved[1::2] = hi
    codes = interleaved[:n_elements] - np.int8(7)
    return codes


def compress(model, calib_tokens, budget_bytes):
    config = model.config.to_dict()
    quant = {}

    GROUP_SIZE = 128
    SEED_BASE = 0xDEADBEEF

    tensors = list(model.state_dict().items())

    for idx, (key, value) in enumerate(tensors):
        arr = value.detach().cpu().float().numpy()

        if arr.ndim < 2 or arr.size < 512:
            quant[key] = ("fp16", arr.astype(np.float16))
            continue

        if "embed" in key.lower() or arr.shape[0] > 50000:
            scale = float(np.abs(arr).max()) / 127.0 or 1.0
            codes = np.clip(np.round(arr / scale), -127, 127).astype(np.int8)
            quant[key] = ("int8", codes, np.float32(scale))
            continue

        orig_shape_full = arr.shape
        W = arr.reshape(-1, arr.shape[-1]) if arr.ndim > 2 else arr
        rows, cols = W.shape

        seed = (SEED_BASE + idx * 7919) & 0xFFFFFFFF
        signs, perm = _make_rotation(cols, seed)

        W_rot = _apply_rotation(W, signs, perm)

        codes, scales, shape_2d, pad, gs = _quantize_int4_pergroup(W_rot, GROUP_SIZE)

        W_rot_hat = _dequantize_int4_pergroup(codes, scales, shape_2d, pad, gs)
        W_hat = _invert_rotation(W_rot_hat, signs, perm)

        mse = float(np.mean((W - W_hat) ** 2))
        var = float(np.var(W)) + 1e-10
        rel_err = mse / var

        if rel_err > 0.08 or not np.isfinite(W_hat).all():
            scale = float(np.abs(W).max()) / 127.0 or 1.0
            codes_i8 = np.clip(np.round(W / scale), -127, 127).astype(np.int8)
            quant[key] = ("int8", codes_i8, np.float32(scale), orig_shape_full)
            continue

        n_elements = codes.size
        packed = _pack_int4(codes)
        scales_bytes = zlib.compress(scales.tobytes(), level=6)

        quant[key] = (
            "rot4", packed, scales_bytes, shape_2d, orig_shape_full,
            pad, gs, seed, n_elements, np.int32(len(scales.ravel())),
        )

    payload = pickle.dumps({"config": config, "quant": quant}, protocol=4)
    return zlib.compress(payload, level=1)


def decompress(artifact):
    payload = zlib.decompress(artifact)
    blob = pickle.loads(payload)

    config = dict(blob["config"])
    model_type = config.pop("model_type")
    cfg = AutoConfig.for_model(model_type, **config)
    model = AutoModelForCausalLM.from_config(cfg)

    state = {}
    for key, record in blob["quant"].items():
        kind = record[0]

        if kind == "fp16":
            arr = record[1].astype(np.float32)
            state[key] = torch.tensor(arr)

        elif kind == "int8":
            codes, scale = record[1], record[2]
            orig_shape_full = record[3] if len(record) > 3 else codes.shape
            arr = codes.astype(np.float32) * float(scale)
            arr = arr.reshape(orig_shape_full)
            state[key] = torch.tensor(arr)

        elif kind == "rot4":
            (_, packed, scales_bytes, shape_2d, orig_shape_full,
             pad, gs, seed, n_elements, n_scales) = record
            codes_i8 = _unpack_int4(packed, int(n_elements))
            scales_raw = zlib.decompress(scales_bytes)
            scales = np.frombuffer(scales_raw, dtype=np.float16).copy()
            scales = scales[:int(n_scales)].reshape(-1, 1)
            W_rot_hat = _dequantize_int4_pergroup(codes_i8, scales, shape_2d, pad, gs)
            signs, perm = _make_rotation(shape_2d[1], int(seed))
            W_hat = _invert_rotation(W_rot_hat, signs, perm)
            W_hat = W_hat.reshape(orig_shape_full).astype(np.float32)
            if not np.isfinite(W_hat).all():
                W_hat = np.nan_to_num(W_hat, nan=0.0, posinf=0.0, neginf=0.0)
            state[key] = torch.tensor(W_hat)
        else:
            raise ValueError(f"Unknown quant kind: {kind}")

    model.load_state_dict(state)
    return model.eval()
