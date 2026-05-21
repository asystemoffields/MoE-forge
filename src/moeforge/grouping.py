"""Cheap, faithful evaluator for carve channel groupings.

A dense gated FFN computes, per intermediate channel i, an activation a[:, i] whose
contribution to the output is the outer product a[:, i] x down[:, i]. Carving partitions
those channels into a shared group (always active) plus E expert groups, of which only
top-k are active per token. A *good* grouping clusters co-activating channels so that few
expert groups reconstruct most of the dense output for any given token.

`oracle_topk_error` measures exactly that: it gives the grouping the benefit of a perfect
router (oracle top-k by per-token contribution) and reports the relative reconstruction
error vs the dense FFN. Lower is better. This decouples *partition quality* from *router
learnability*, so it is a fast surrogate for "is this a good carve grouping" that an
allocation/evolution loop can call thousands of times.
"""

from __future__ import annotations

import numpy as np

SHARED = -1  # assignment value marking an always-active (shared) channel.


def silu(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-x))


def intermediate_activations(hidden: np.ndarray, gate: np.ndarray, up: np.ndarray) -> np.ndarray:
    """Per-token, per-channel gated activation a = silu(h@gate^T) * (h@up^T). Shape [T, I]."""
    return silu(hidden @ gate.T) * (hidden @ up.T)


def oracle_topk_error(
    *,
    activations: np.ndarray,
    down: np.ndarray,
    assignment: np.ndarray,
    top_k: int,
) -> float:
    """Relative reconstruction error of shared + oracle-top-k expert groups vs the dense FFN.

    activations: [T, I] gated activations. down: [H, I]. assignment: [I] ints, SHARED for
    always-active channels, else a non-negative expert id. Returns mean over tokens of
    ||dense - reconstruction|| / ||dense||.
    """
    activations = np.asarray(activations, dtype=np.float64)
    down = np.asarray(down, dtype=np.float64)
    assignment = np.asarray(assignment)
    if activations.shape[1] != down.shape[1] or assignment.shape[0] != down.shape[1]:
        raise ValueError("activations, down, and assignment must agree on the channel dimension")

    dense = activations @ down.T
    shared_mask = assignment == SHARED
    reconstruction = (
        activations[:, shared_mask] @ down[:, shared_mask].T
        if shared_mask.any()
        else np.zeros_like(dense)
    )

    expert_ids = sorted({int(value) for value in assignment if value != SHARED})
    if expert_ids:
        token_count = activations.shape[0]
        contributions: list[np.ndarray] = []
        norms = np.zeros((token_count, len(expert_ids)))
        for column, expert in enumerate(expert_ids):
            mask = assignment == expert
            contribution = activations[:, mask] @ down[:, mask].T
            contributions.append(contribution)
            norms[:, column] = np.linalg.norm(contribution, axis=1)
        k = min(int(top_k), len(expert_ids))
        selected = np.argsort(-norms, axis=1)[:, :k]
        for column in range(len(expert_ids)):
            chosen = (selected == column).any(axis=1)
            if chosen.any():
                reconstruction[chosen] += contributions[column][chosen]

    error = np.linalg.norm(dense - reconstruction, axis=1) / (np.linalg.norm(dense, axis=1) + 1e-12)
    return float(error.mean())


def channel_importance(activations: np.ndarray) -> np.ndarray:
    """Per-channel mean absolute gated activation — the signal the magnitude oracle uses."""
    return np.abs(np.asarray(activations, dtype=np.float64)).mean(axis=0)


def magnitude_grouping(
    importance: np.ndarray,
    *,
    n_experts: int,
    shared_ratio: float,
) -> np.ndarray:
    """Baseline: most-important channels become shared; the rest round-robin into experts
    in importance order (so each expert gets a balanced spread of importance)."""
    importance = np.asarray(importance, dtype=np.float64)
    channel_count = importance.shape[0]
    n_shared = int(round(shared_ratio * channel_count))
    order = np.argsort(-importance)
    assignment = np.empty(channel_count, dtype=int)
    assignment[order[:n_shared]] = SHARED
    for position, channel in enumerate(order[n_shared:]):
        assignment[channel] = position % max(1, n_experts)
    return assignment


def random_grouping(
    channel_count: int,
    *,
    n_experts: int,
    shared_ratio: float,
    rng: np.random.Generator,
) -> np.ndarray:
    n_shared = int(round(shared_ratio * channel_count))
    assignment = rng.integers(0, max(1, n_experts), size=channel_count)
    shared = rng.choice(channel_count, size=n_shared, replace=False)
    assignment[shared] = SHARED
    return assignment


def balanced_assign(features: np.ndarray, k: int, *, rng: np.random.Generator, iters: int = 12) -> np.ndarray:
    """Equal-size clustering: like k-means but each cluster is capped at ceil(n/k) members.
    Greedy capacity assignment ordered by per-point regret (decisiveness)."""
    n = features.shape[0]
    cap = -(-n // max(1, k))  # ceil
    centers = features[rng.choice(n, min(k, n), replace=False)].astype(np.float64).copy()
    labels = np.zeros(n, dtype=int)
    for _ in range(iters):
        dist = ((features[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)  # [n, k]
        order = np.argsort(dist, axis=1)
        best = dist[np.arange(n), order[:, 0]]
        second = dist[np.arange(n), order[:, 1]] if k > 1 else best
        proc = np.argsort(-(second - best))  # most-decisive points first
        labels = np.full(n, -1, dtype=int)
        counts = np.zeros(k, dtype=int)
        for point in proc:
            for cluster in order[point]:
                if counts[cluster] < cap:
                    labels[point] = cluster
                    counts[cluster] += 1
                    break
        for cluster in range(k):
            members = features[labels == cluster]
            if len(members):
                centers[cluster] = members.mean(axis=0)
    return labels


def balanced_grouping(
    activations: np.ndarray,
    importance: np.ndarray,
    *,
    n_experts: int,
    shared_ratio: float,
    rng: np.random.Generator,
    transform: str = "abs",
) -> np.ndarray:
    """Shared = top-importance channels; the rest split into EQUAL-SIZE experts by clustering
    their (transformed) activation vectors. transform: 'raw' | 'abs' | 'squared'. Equal sizes
    fix the active fraction at a given top_k, so this isolates grouping quality from budget-edge
    effects. 'abs'/'squared' capture co-firing under the gated FFN (opposite-sign partners,
    energy/contribution proxy)."""
    importance = np.asarray(importance, dtype=np.float64)
    activations = np.asarray(activations, dtype=np.float64)
    channel_count = importance.shape[0]
    n_shared = int(round(shared_ratio * channel_count))
    order = np.argsort(-importance)
    assignment = np.full(channel_count, SHARED, dtype=int)
    remaining = order[n_shared:]
    vectors = activations[:, remaining].T
    if transform == "abs":
        vectors = np.abs(vectors)
    elif transform == "squared":
        vectors = vectors ** 2
    vectors = vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-12)
    labels = balanced_assign(vectors, n_experts, rng=rng)
    for channel, label in zip(remaining, labels):
        assignment[channel] = int(label)
    return assignment
