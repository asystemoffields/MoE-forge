"""gen1_whiten: whiten the hidden state before matching, plus a per-expert base-rate bias.

Raw dot products are dominated by high-variance hidden dimensions, which may not be the ones
that discriminate experts. Standardize the hidden state with calibration mean/std, build each
expert key as the mean whitened hidden over the tokens where it is truly hot, and add a bias =
log(hot rate) so experts that fire often are not unfairly suppressed."""

import numpy as np

SHARED = -1


def build_router(ctx, n_experts, top_k, rng):
    hidden = ctx["calib_hidden"]
    activations = ctx["calib_activations"]
    down = ctx["down"]
    assignment = ctx["assignment"]

    mean = hidden.mean(axis=0)
    std = hidden.std(axis=0) + 1e-6
    whitened = (hidden - mean) / std

    energy = np.zeros((hidden.shape[0], n_experts))
    for expert in range(n_experts):
        mask = assignment == expert
        if mask.any():
            energy[:, expert] = np.linalg.norm(activations[:, mask] @ down[:, mask].T, axis=1)
    threshold = np.sort(energy, axis=1)[:, -top_k]

    keys = np.zeros((n_experts, hidden.shape[1]))
    bias = np.zeros(n_experts)
    for expert in range(n_experts):
        hot = energy[:, expert] >= threshold
        if hot.any():
            keys[expert] = whitened[hot].mean(axis=0)
        bias[expert] = np.log(hot.mean() + 1e-6)
    return {"mean": mean, "std": std, "keys": keys, "bias": bias}


def route(hidden, state, n_experts, top_k):
    whitened = (hidden - state["mean"]) / state["std"]
    return whitened @ state["keys"].T + state["bias"]
