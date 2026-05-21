"""Seed router: each expert gets one key = importance-weighted mean of its channels' gate
rows, normalized. Route by dot product of the hidden state with each expert key.

Rationale: expert e's channels produce large pre-activations when the hidden state h aligns
with their gate rows, so the (importance-weighted) mean gate direction is a cheap, training-
free predictor of "is expert e active for this token". This is the naive realizable router;
the oracle floor knows each expert's TRUE output norm. The gap the search can close: the
silu nonlinearity, the multiplicative up-projection, down-row magnitudes (which channels
actually matter in the output), expert multimodality (one key is too few), and predicting
contribution MAGNITUDE rather than just direction. Calibration tokens (ctx["calib_*"]) are
available for a closed-form fit but unused here."""

import numpy as np

SHARED = -1


def build_router(ctx, n_experts, top_k, rng):
    gate = ctx["gate"]
    assignment = ctx["assignment"]
    importance = ctx["importance"]
    hidden_dim = gate.shape[1]
    keys = np.zeros((n_experts, hidden_dim))
    for expert in range(n_experts):
        mask = assignment == expert
        if not mask.any():
            continue
        weight = importance[mask]
        key = (gate[mask] * weight[:, None]).sum(axis=0) / (weight.sum() + 1e-12)
        keys[expert] = key / (np.linalg.norm(key) + 1e-12)
    return {"keys": keys}


def route(hidden, state, n_experts, top_k):
    return hidden @ state["keys"].T
