"""gen1_multikey: two keys per expert to capture multimodal experts.

One key per expert is weak when an expert's channels co-fire in several distinct hidden-state
regimes. On the calibration split, find the tokens where each expert is truly in the top-k,
cluster those hidden states into two centroids, and use both as keys. Score = max over the
expert's keys of the dot product, so an expert wins if the token matches ANY of its modes."""

import numpy as np

SHARED = -1
N_KEYS = 2


def _kmeans(points, k, rng, iters=15):
    if len(points) <= k:
        centers = np.zeros((k, points.shape[1]))
        centers[: len(points)] = points
        return centers
    centers = points[rng.choice(len(points), k, replace=False)].copy()
    for _ in range(iters):
        dist = ((points[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        labels = dist.argmin(axis=1)
        for cluster in range(k):
            members = points[labels == cluster]
            if len(members):
                centers[cluster] = members.mean(axis=0)
    return centers


def build_router(ctx, n_experts, top_k, rng):
    hidden = ctx["calib_hidden"]
    activations = ctx["calib_activations"]
    down = ctx["down"]
    assignment = ctx["assignment"]
    hidden_dim = hidden.shape[1]

    energy = np.zeros((hidden.shape[0], n_experts))
    for expert in range(n_experts):
        mask = assignment == expert
        if mask.any():
            energy[:, expert] = np.linalg.norm(activations[:, mask] @ down[:, mask].T, axis=1)
    threshold = np.sort(energy, axis=1)[:, -top_k]  # kth-largest per token

    keys = np.zeros((n_experts, N_KEYS, hidden_dim))
    for expert in range(n_experts):
        hot = energy[:, expert] >= threshold
        points = hidden[hot] if int(hot.sum()) >= N_KEYS else hidden
        centers = _kmeans(points, N_KEYS, rng)
        for j in range(N_KEYS):
            keys[expert, j] = centers[j] / (np.linalg.norm(centers[j]) + 1e-12)
    return {"keys": keys.reshape(n_experts * N_KEYS, hidden_dim), "n_keys": np.array([N_KEYS])}


def route(hidden, state, n_experts, top_k):
    n_keys = int(state["n_keys"][0])
    sims = hidden @ state["keys"].T  # [T, E*n_keys]
    return sims.reshape(hidden.shape[0], n_experts, n_keys).max(axis=2)
