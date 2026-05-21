"""Score carve channel-grouping strategies against the oracle-top-k reconstruction metric.

This is the testbed for closed-loop ("DIY AlphaEvolve") search: a candidate is a *grouping
function* `group(ctx, n_experts, shared_ratio, rng) -> assignment`, scored by the mean
oracle-top-k reconstruction error across captured layers (lower = better). Baselines here are
the magnitude oracle (current default), random, and co-activation k-means clustering (the
MoEfication idea). A per-layer hill-climb shows how much headroom the metric exposes.

The point: an evolution loop can mutate the *code* of `group` and keep whatever lowers the
score — and we can check whether the evolved heuristic beats the baselines on held-out layers.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from moeforge.grouping import SHARED, magnitude_grouping, oracle_topk_error, random_grouping


def load_layers(paths: list[Path]) -> list[dict]:
    layers = []
    for path in paths:
        data = np.load(path)
        layers.append(
            {
                "name": path.stem,
                "activations": data["activations"].astype(np.float64),
                "down": data["down"].astype(np.float64),
                "importance": data["importance"].astype(np.float64),
            }
        )
    return layers


# ---- candidate grouping functions (the unit an evolution loop would mutate) ----

def group_magnitude(ctx, n_experts, shared_ratio, rng):
    return magnitude_grouping(ctx["importance"], n_experts=n_experts, shared_ratio=shared_ratio)


def group_random(ctx, n_experts, shared_ratio, rng):
    return random_grouping(ctx["importance"].shape[0], n_experts=n_experts, shared_ratio=shared_ratio, rng=rng)


def group_coactivation_kmeans(ctx, n_experts, shared_ratio, rng):
    """Shared = top-importance channels; the rest clustered by co-activation pattern (cosine
    k-means on channel activation vectors), so channels that fire together share an expert."""
    importance = ctx["importance"]
    activations = ctx["activations"]
    channel_count = importance.shape[0]
    n_shared = int(round(shared_ratio * channel_count))
    order = np.argsort(-importance)
    assignment = np.empty(channel_count, dtype=int)
    assignment[order[:n_shared]] = SHARED
    remaining = order[n_shared:]
    vectors = activations[:, remaining].T  # [n_remaining, T]
    vectors = vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-12)
    labels = _kmeans(vectors, n_experts, rng)
    for channel, label in zip(remaining, labels):
        assignment[channel] = int(label)
    return assignment


def _kmeans(points: np.ndarray, k: int, rng, iters: int = 25) -> np.ndarray:
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


# ---- scoring ----

def score_function(fn, layers, *, n_experts, shared_ratio, top_k, seed=0) -> tuple[float, list[float]]:
    rng = np.random.default_rng(seed)
    errors = []
    for layer in layers:
        ctx = {"importance": layer["importance"], "activations": layer["activations"]}
        assignment = fn(ctx, n_experts, shared_ratio, rng)
        errors.append(
            oracle_topk_error(
                activations=layer["activations"], down=layer["down"], assignment=assignment, top_k=top_k
            )
        )
    return float(np.mean(errors)), errors


def hill_climb(layer, *, n_experts, shared_ratio, top_k, iters, seed=0) -> float:
    """Per-layer optimization ceiling: mutate the assignment, keep improvements."""
    rng = np.random.default_rng(seed)
    ctx = {"importance": layer["importance"], "activations": layer["activations"]}
    assignment = magnitude_grouping(layer["importance"], n_experts=n_experts, shared_ratio=shared_ratio)
    best = oracle_topk_error(activations=layer["activations"], down=layer["down"], assignment=assignment, top_k=top_k)
    n = assignment.shape[0]
    for _ in range(iters):
        channel = int(rng.integers(0, n))
        old = assignment[channel]
        assignment[channel] = int(rng.integers(0, n_experts))
        err = oracle_topk_error(activations=layer["activations"], down=layer["down"], assignment=assignment, top_k=top_k)
        if err < best:
            best = err
        else:
            assignment[channel] = old
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layers", type=Path, nargs="+", required=True, help="layer .npz files from capture_layer.py")
    parser.add_argument("--experts", type=int, default=8)
    parser.add_argument("--shared-ratio", type=float, default=0.125)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--hill-climb-iters", type=int, default=1500)
    args = parser.parse_args()

    layers = load_layers(args.layers)
    regime = dict(n_experts=args.experts, shared_ratio=args.shared_ratio, top_k=args.top_k)
    print(f"regime: experts={args.experts} shared_ratio={args.shared_ratio} top_k={args.top_k}")
    print(f"layers: {[l['name'] for l in layers]}\n")

    print(f"{'strategy':<26}{'mean error':>12}   per-layer")
    candidates = [
        ("random (best of 8)", _best_random(layers, regime)),
        ("magnitude oracle", score_function(group_magnitude, layers, **regime)),
        ("co-activation kmeans", score_function(group_coactivation_kmeans, layers, **regime)),
    ]
    for name, (mean_err, per_layer) in candidates:
        per = " ".join(f"{e:.4f}" for e in per_layer)
        print(f"{name:<26}{mean_err:>12.4f}   {per}")

    ceilings = [hill_climb(layer, iters=args.hill_climb_iters, **regime) for layer in layers]
    print(f"{'hill-climb (per-layer)':<26}{np.mean(ceilings):>12.4f}   " + " ".join(f"{e:.4f}" for e in ceilings))


def _best_random(layers, regime):
    best_mean = None
    best_per = None
    for seed in range(8):
        mean_err, per = score_function(group_random, layers, seed=seed, **regime)
        if best_mean is None or mean_err < best_mean:
            best_mean, best_per = mean_err, per
    return best_mean, best_per


if __name__ == "__main__":
    main()
