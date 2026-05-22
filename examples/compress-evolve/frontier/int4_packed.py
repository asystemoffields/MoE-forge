"""Seed: compute-in-compressed-form int4. The decompressed model HOLDS int4-packed Linear weights
(uint8 buffers) and dequantizes per layer in forward, so it runs in ~int4 RESIDENT memory rather
than inflating to fp32. Non-matmul params (embeddings, norms) kept at bf16. This is the seed for
the resident-memory axis -- it actually runs small, unlike the disk-only int8 seed."""

FAMILY = "int4_packed"

import pickle
import sys
import zlib
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import packed as P

GROUP = 128


def compress(model, calib_tokens, budget_bytes):
    cfg = model.config.to_dict()
    sd = model.state_dict()
    linear_owners = {n for n, m in model.named_modules() if isinstance(m, nn.Linear)}

    linears, other = {}, {}
    handled = set()
    for owner in linear_owners:
        wk, bk = owner + ".weight", owner + ".bias"
        if wk in sd and sd[wk].ndim == 2:
            pk, sc, in_f = P.pack_int4_grouped(sd[wk].float(), GROUP)
            b = sd[bk].cpu().to(torch.float16).numpy() if bk in sd else None
            linears[owner] = (pk.numpy(), sc.numpy(), int(sd[wk].shape[0]), int(in_f), b)
            handled.add(wk)
            handled.add(bk)
    for k, v in sd.items():
        if k not in handled:
            other[k] = v.cpu().to(torch.float16).numpy()

    blob = {"config": cfg, "linears": linears, "other": other, "group": GROUP}
    return zlib.compress(pickle.dumps(blob, protocol=4), level=6)


def decompress(artifact):
    blob = pickle.loads(zlib.decompress(artifact))
    cfg = dict(blob["config"])
    model_type = cfg.pop("model_type")
    model = AutoModelForCausalLM.from_config(AutoConfig.for_model(model_type, **cfg))
    group = blob["group"]
    linears = blob["linears"]

    def build(full, lin):
        if full in linears:
            pk, sc, out_f, in_f, b = linears[full]
            bias = torch.from_numpy(b.astype(np.float16)) if b is not None else None
            return P.Int4Linear(torch.from_numpy(pk), torch.from_numpy(sc), out_f, in_f, bias, group)
        return lin

    P.replace_linears(model, build)

    msd = model.state_dict()
    to_load = {k: torch.from_numpy(v.astype(np.float32)) for k, v in blob["other"].items() if k in msd}
    model.load_state_dict(to_load, strict=False)
    return model.to(torch.bfloat16).eval()
