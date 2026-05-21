"""Test: activation-only local refinement without using 'down' matrix.

Proxy: for each expert, score[t, e] = sum_{i in e} act[t,i]^2 (squared activation sum).
This approximates the expert's contribution norm when down columns have similar magnitudes.
We greedily move channels to maximize top-k coverage of this proxy score.
"""
import sys, time
sys.path.insert(0, 'src')
import numpy as np
from moeforge.grouping import oracle_topk_error, SHARED

import importlib.util
spec = importlib.util.spec_from_file_location('c', 'examples/grouping-search/candidates/seed_clustering.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def act_sweep(act, assign, n_experts, top_k, rng_seed):
    """One sweep of activation-proxy-based local refinement."""
    T, I = act.shape
    act_sq = act ** 2  # [T, I]

    # Expert activation-squared sums: [E, T]
    expert_sq = np.zeros((n_experts, T))
    for e in range(n_experts):
        mask = assign == e
        if mask.any():
            expert_sq[e] = act_sq[:, mask].sum(axis=1)

    non_shared = np.where(assign != SHARED)[0]
    order = np.random.default_rng(rng_seed).permutation(non_shared)
    n_moves = 0

    # Current top-k coverage
    cur_topk = np.sort(expert_sq, axis=0)[-top_k:, :].sum()

    for i in order:
        src = int(assign[i])
        sq_i = act_sq[:, i]   # [T] -- squared activation of channel i

        # New expert_sq for src (after removing i)
        new_sq_src = expert_sq[src] - sq_i  # [T]

        # For each dst: new expert_sq[dst] = expert_sq[dst] + sq_i
        # Try all dsts and find best
        best_cov = cur_topk
        best_dst = src

        for dst in range(n_experts):
            if dst == src:
                continue
            # Build trial expert_sq (only src and dst change)
            trial = expert_sq.copy()
            trial[src] = new_sq_src
            trial[dst] = expert_sq[dst] + sq_i
            # Top-k coverage
            topk = np.sort(trial, axis=0)[-top_k:, :].sum()
            if topk > best_cov:
                best_cov = topk
                best_dst = dst

        if best_dst != src:
            expert_sq[src] = new_sq_src
            expert_sq[best_dst] = expert_sq[best_dst] + sq_i
            cur_topk = best_cov
            assign[i] = best_dst
            n_moves += 1

    return n_moves, assign, expert_sq


def refine_actonly(ctx, n_experts, shared_ratio, rng, top_k=2, n_sweeps=20, time_limit=20):
    importance = ctx["importance"]
    activations = ctx["activations"]
    T, I = activations.shape

    # Build seed assignment
    n_shared = int(round(shared_ratio * I))
    order = np.argsort(-importance)
    assignment = np.empty(I, dtype=int)
    assignment[order[:n_shared]] = SHARED
    remaining = order[n_shared:]
    vectors = activations[:, remaining].T
    vectors = vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-12)

    # K-means
    k = n_experts
    centers = vectors[rng.choice(len(remaining), k, replace=False)].copy()
    labels = np.zeros(len(remaining), dtype=int)
    for _ in range(25):
        distances = ((vectors[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        labels = distances.argmin(axis=1)
        for cluster in range(k):
            members = vectors[labels == cluster]
            if len(members):
                centers[cluster] = members.mean(axis=0)
    for channel, label in zip(remaining, labels):
        assignment[channel] = int(label)

    t_start = time.time()
    for sweep in range(n_sweeps):
        n_moves, assignment, _ = act_sweep(activations, assignment, n_experts, top_k, sweep)
        if n_moves == 0:
            break
        if time.time() - t_start > time_limit:
            break

    return assignment


# Test on both layers
for layer_name in ['layer3', 'layer9']:
    print(f'\n=== {layer_name} ===')
    data = np.load(f'examples/grouping-search/{layer_name}.npz')
    act = data['activations'].astype(np.float64)
    imp = data['importance'].astype(np.float64)
    down = data['down'].astype(np.float64)

    rng = np.random.default_rng(0)
    ctx = {'importance': imp, 'activations': act}

    t0 = time.time()
    assign = refine_actonly(ctx, 8, 0.125, rng, n_sweeps=30, time_limit=20)
    dt = time.time() - t0

    err = oracle_topk_error(activations=act, down=down, assignment=assign, top_k=2)
    print(f'  Oracle error: {err:.6f}, time: {dt:.1f}s')

    # Check balance
    routed = assign[assign != SHARED]
    counts = np.bincount(routed, minlength=8)
    print(f'  Expert sizes: {counts.tolist()}, max/ideal = {counts.max()/(routed.size/8):.2f}')
