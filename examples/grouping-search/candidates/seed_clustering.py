"""Seed candidate: shared = top-importance channels; rest clustered by co-activation
(cosine k-means on channel activation vectors). This is the current best baseline."""

import numpy as np

SHARED = -1


def _kmeans(points, k, rng, iters=25):
    n = points.shape[0]
    centers = points[rng.choice(n, k, replace=False)].copy()
    labels = np.zeros(n, dtype=int)
    for _ in range(iters):
        distances = ((points[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        labels = distances.argmin(axis=1)
        for cluster in range(k):
            members = points[labels == cluster]
            if len(members):
                centers[cluster] = members.mean(axis=0)
    return labels


def group(ctx, n_experts, shared_ratio, rng):
    importance = ctx["importance"]
    activations = ctx["activations"]
    channel_count = importance.shape[0]
    n_shared = int(round(shared_ratio * channel_count))
    order = np.argsort(-importance)
    assignment = np.empty(channel_count, dtype=int)
    assignment[order[:n_shared]] = SHARED
    remaining = order[n_shared:]
    vectors = activations[:, remaining].T
    vectors = vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-12)
    labels = _kmeans(vectors, n_experts, rng)
    for channel, label in zip(remaining, labels):
        assignment[channel] = int(label)
    return assignment
