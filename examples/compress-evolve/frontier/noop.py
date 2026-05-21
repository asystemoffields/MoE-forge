"""Seed: lossless no-op. Serializes weights + config exactly (torch.save handles any dtype,
incl. bf16); reconstructs bit-for-bit. The full-size, zero-distortion corner of the
rate-distortion frontier (ratio ~1x, nll == baseline)."""

import io

import torch
from transformers import AutoConfig, AutoModelForCausalLM


def compress(model, calib_tokens, budget_bytes):
    buffer = io.BytesIO()
    torch.save({"config": model.config.to_dict(), "state": model.state_dict()}, buffer)
    return buffer.getvalue()


def decompress(artifact):
    blob = torch.load(io.BytesIO(artifact), weights_only=False)
    config = dict(blob["config"])
    model_type = config.pop("model_type")
    cfg = AutoConfig.for_model(model_type, **config)
    model = AutoModelForCausalLM.from_config(cfg)
    model.load_state_dict(blob["state"])
    return model.eval()
