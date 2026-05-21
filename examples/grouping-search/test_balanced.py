"""Test: balanced activation-only local refinement.

Now with balance constraint: no expert can have more than 2x ideal size.
During refinement, we only allow moves that keep both src and dst in bounds.
"""
import sys, time
sys.path.insert(0, 'src')
import numpy as np
from moeforge.grouping import oracle_topk_error, SHARED

import importlib.util
spec = importlib.util.spec_from_file_location('c', 'examples/grouping-search/candidates/seed_clustering.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def act_sweep_balanced(act, assign, n_experts, top_k, rng_seed, max_size):
    """One sweep, respecting max_size per expert."""
    T, I = act.shape
    act_sq = act ** 2  # [T, I]

    # Expert activation-squared sums: [E, T]
    expert_sq = np.zeros((n_experts, T))
    expert_size = np.zeros(n_experts, dtype=int)
    for e in range(n_experts):
        mask = assign == e
        expert_size[e] = mask.sum()
        if mask.any():
            expert_sq[e] = act_sq[:, mask].sum(axis=1)

    non_shared = np.where(assign != SHARED)[0]
    order = np.random.default_rng(rng_seed).permutation(non_shared)
    n_moves = 0

    cur_topk = np.sort(expert_sq, axis=0)[-top_k:, :].sum()

    for i in order:
        src = int(assign[i])
        sq_i = act_sq[:, i]

        new_sq_src = expert_sq[src] - sq_i

        best_cov = cur_topk
        best_dst = src

        for dst in range(n_experts):
            if dst == src:
                continue
            # Balance check: dst must not exceed max_size
            if expert_size[dst] >= max_size:
                continue
            # Src must keep at least 1 channel (avoid emptying experts)
            if expert_size[src] <= 1:
                continue

            trial = expert_sq.copy()
            trial[src] = new_sq_src
            trial[dst] = expert_sq[dst] + sq_i
            topk = np.sort(trial, axis=0)[-top_k:, :].sum()
            if topk > best_cov:
                best_cov = topk
                best_dst = dst

        if best_dst != src:
            expert_sq[src] = new_sq_src
            expert_sq[best_dst] = expert_sq[best_dst] + sq_i
            expert_size[src] -= 1
            expert_size[best_dst] += 1
            cur_topk = best_cov
            assign[i] = best_dst
            n_moves += 1

    return n_moves, assign, expert_sq, expert_size


def refine_actonly_balanced(ctx, n_experts, shared_ratio, rng, top_k=2, n_sweeps=30, time_limit=25):
    importance = ctx["importance"]
    activations = ctx["activations"]
    T, I = activations.shape
    n_shared = int(round(shared_ratio * I))
    n_routed = I - n_shared
    ideal = n_routed / n_experts
    max_size = int(2.0 * ideal)  # balance cap from eval_candidate.py

    # Build seed with k-means
    order = np.argsort(-importance)
    assignment = np.empty(I, dtype=int)
    assignment[order[:n_shared]] = SHARED
    remaining = order[n_shared:]
    vectors = activations[:, remaining].T
    vectors = vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-12)

    centers = vectors[rng.choice(len(remaining), n_experts, replace=False)].copy()
    labels = np.zeros(len(remaining), dtype=int)
    for _ in range(25):
        distances = ((vectors[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        labels = distances.argmin(axis=1)
        for cluster in range(n_experts):
            members = vectors[labels == cluster]
            if len(members):
                centers[cluster] = members.mean(axis=0)
    for channel, label in zip(remaining, labels):
        assignment[channel] = int(label)

    t_start = time.time()
    for sweep in range(n_sweeps):
        n_moves, assignment, _, _ = act_sweep_balanced(
            activations, assignment, n_experts, top_k, sweep, max_size)
        if n_moves == 0:
            break
        if time.time() - t_start > time_limit:
            break

    return assignment


# Test on both layers
baseline_errors = {'layer3': 0.560670, 'layer9': 0.570832}

for layer_name in ['layer3', 'layer9']:
    print(f'\n=== {layer_name} ===')
    data = np.load(f'examples/grouping-search/{layer_name}.npz')
    act = data['activations'].astype(np.float64)
    imp = data['importance'].astype(np.float64)
    down = data['down'].astype(np.float64)

    rng = np.random.default_rng(0)
    ctx = {'importance': imp, 'activations': act}

    t0 = time.time()
    assign = refine_actonly_balanced(ctx, 8, 0.125, rng, n_sweeps=30, time_limit=25)
    dt = time.time() - t0

    err = oracle_topk_error(activations=act, down=down, assignment=assign, top_k=2)
    baseline = baseline_errors[layer_name]
    print(f'  Oracle error: {err:.6f} (baseline: {baseline:.6f}, improvement: {(baseline-err)/baseline*100:.1f}%)')
    print(f'  Time: {dt:.1f}s')

    routed = assign[assign != SHARED]
    counts = np.bincount(routed, minlength=8)
    ideal = len(routed) / 8
    print(f'  Expert sizes: {counts.tolist()}, max/ideal = {counts.max()/ideal:.2f}')
    print(f'  All experts used: {(counts > 0).all()}')
    print(f'  Satisfies balance (<= 2x ideal = {int(2*ideal)}): {counts.max() <= int(2*ideal)}')
