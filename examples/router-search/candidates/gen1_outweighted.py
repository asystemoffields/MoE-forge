"""gen1_outweighted: like the seed, but weight each channel's gate row by how much it actually
matters in the OUTPUT, not just its activation magnitude.

A channel's contribution to the FFN output is a[:, i] * down[:, i], so a channel with a large
||down[:, i]|| dominates the output even at modest activation. The seed weights gate rows by
activation importance alone; here the per-channel weight is importance * ||down[:, i]||, so the
expert key points toward the hidden directions that drive the output the router is judged on."""

import numpy as np

SHARED = -1


def build_router(ctx, n_experts, top_k, rng):
    gate = ctx["gate"]
    down = ctx["down"]
    assignment = ctx["assignment"]
    importance = ctx["importance"]
    hidden_dim = gate.shape[1]

    down_norm = np.linalg.norm(down, axis=0)  # [I] per-channel output relevance
    keys = np.zeros((n_experts, hidden_dim))
    for expert in range(n_experts):
        mask = assignment == expert
        if not mask.any():
            continue
        weight = importance[mask] * down_norm[mask]
        key = (gate[mask] * weight[:, None]).sum(axis=0) / (weight.sum() + 1e-12)
        keys[expert] = key / (np.linalg.norm(key) + 1e-12)
    return {"keys": keys}


def route(hidden, state, n_experts, top_k):
    return hidden @ state["keys"].T
