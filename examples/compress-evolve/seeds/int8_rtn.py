"""Seed: per-tensor symmetric int8 round-to-nearest on matrices (the textbook PTQ baseline).
Matrices -> int8 + a scale; 1-D params (norms, biases) kept fp16. ~3.8x smaller, small NLL hit."""

import pickle

import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM


def compress(model, calib_tokens, budget_bytes):
    config = model.config.to_dict()
    quant = {}
    for key, value in model.state_dict().items():
        array = value.detach().cpu().float().numpy()
        if array.ndim >= 2:
            scale = float(np.abs(array).max()) / 127.0 or 1.0
            codes = np.clip(np.round(array / scale), -127, 127).astype(np.int8)
            quant[key] = ("int8", codes, np.float32(scale))
        else:
            quant[key] = ("fp16", array.astype(np.float16))
    return pickle.dumps({"config": config, "quant": quant}, protocol=4)


def decompress(artifact):
    blob = pickle.loads(artifact)
    config = dict(blob["config"])
    model_type = config.pop("model_type")
    cfg = AutoConfig.for_model(model_type, **config)
    model = AutoModelForCausalLM.from_config(cfg)
    state = {}
    for key, record in blob["quant"].items():
        if record[0] == "int8":
            _, codes, scale = record
            state[key] = torch.tensor(codes.astype(np.float32) * float(scale))
        else:
            state[key] = torch.tensor(record[1].astype(np.float32))
    model.load_state_dict(state)
    return model.eval()
