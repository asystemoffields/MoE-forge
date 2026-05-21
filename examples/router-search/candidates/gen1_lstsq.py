"""gen1_lstsq: fit the router in closed form to predict each expert's true output norm.

The oracle ranks experts by ||contribution_e||. So on the calibration split, compute that
target per expert and ridge-regress it from the hidden state: a linear map hidden -> expert
energy. route() then just applies that map. This is a training-free warm-started router that
directly approximates the oracle's selection signal, not merely gate-direction alignment."""

import numpy as np

SHARED = -1


def build_router(ctx, n_experts, top_k, rng):
    hidden = ctx["calib_hidden"]
    activations = ctx["calib_activations"]
    down = ctx["down"]
    assignment = ctx["assignment"]
    tokens, hidden_dim = hidden.shape

    targets = np.zeros((tokens, n_experts))
    for expert in range(n_experts):
        mask = assignment == expert
        if mask.any():
            contribution = activations[:, mask] @ down[:, mask].T
            targets[:, expert] = np.linalg.norm(contribution, axis=1)

    design = np.concatenate([hidden, np.ones((tokens, 1))], axis=1)  # bias column
    ridge = 1e-2 * tokens * np.eye(hidden_dim + 1)
    weights = np.linalg.solve(design.T @ design + ridge, design.T @ targets)  # [H+1, E]
    return {"weights": weights}


def route(hidden, state, n_experts, top_k):
    design = np.concatenate([hidden, np.ones((hidden.shape[0], 1))], axis=1)
    return design @ state["weights"]
