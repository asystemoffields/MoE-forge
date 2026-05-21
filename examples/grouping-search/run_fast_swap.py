"""Test fast swap implementation using incremental top-k updates."""
import sys, time
sys.path.insert(0, 'src')
import numpy as np
from moeforge.grouping import oracle_topk_error

SHARED = -1


def _kmeans(points, k, rng, iters=25):
    n = points.shape[0]
    centers = points[rng.choice(n, k, replace=False)].copy()
    labels = np.zeros(n, dtype=int)
    for _ in range(iters):
        distances = ((points[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        labels = distances.argmin(axis=1)
        for c in range(k):
            members = points[labels == c]
            if len(members):
                centers[c] = members.mean(axis=0)
    return labels


def _make_seed(activations, importance, n_experts, shared_ratio, rng):
    I = importance.shape[0]
    n_shared = int(round(shared_ratio * I))
    order = np.argsort(-importance)
    assignment = np.empty(I, dtype=int)
    assignment[order[:n_shared]] = SHARED
    remaining = order[n_shared:]
    vecs = activations[:, remaining].T
    vecs = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12)
    labels = _kmeans(vecs, n_experts, rng)
    for ch, lab in zip(remaining, labels):
        assignment[ch] = int(lab)
    return assignment


def _sweep(activations, assignment, n_experts, top_k, rng_seed, max_size):
    T, I = activations.shape
    act_sq = activations ** 2
    exp_sq = np.zeros((n_experts, T))
    exp_sz = np.zeros(n_experts, dtype=int)
    for e in range(n_experts):
        mask = assignment == e
        exp_sz[e] = int(mask.sum())
        if mask.any():
            exp_sq[e] = act_sq[:, mask].sum(axis=1)
    non_shared = np.where(assignment != SHARED)[0]
    order = np.random.default_rng(rng_seed).permutation(non_shared)
    n_moves = 0
    cur_cov = np.sort(exp_sq, axis=0)[-top_k:, :].sum()
    for i in order:
        src = int(assignment[i])
        sq_i = act_sq[:, i]
        new_sq_src = exp_sq[src] - sq_i
        best_cov = cur_cov
        best_dst = src
        for dst in range(n_experts):
            if dst == src:
                continue
            if exp_sz[dst] >= max_size:
                continue
            if exp_sz[src] <= 1:
                continue
            trial = exp_sq.copy()
            trial[src] = new_sq_src
            trial[dst] = exp_sq[dst] + sq_i
            cov = np.sort(trial, axis=0)[-top_k:, :].sum()
            if cov > best_cov:
                best_cov = cov
                best_dst = dst
        if best_dst != src:
            exp_sq[src] = new_sq_src
            exp_sq[best_dst] = exp_sq[best_dst] + sq_i
            exp_sz[src] -= 1
            exp_sz[best_dst] += 1
            cur_cov = best_cov
            assignment[i] = best_dst
            n_moves += 1
    return n_moves, assignment, cur_cov


def _swap_sweep_fast(activations, assignment, n_experts, top_k, rng_seed, n_candidates):
    """Faster swap sweep using precomputed sorted indices for O(n_experts) proxy update."""
    T, I = activations.shape
    act_sq = activations ** 2

    exp_sq = np.zeros((n_experts, T))
    for e in range(n_experts):
        mask = assignment == e
        if mask.any():
            exp_sq[e] = act_sq[:, mask].sum(axis=1)

    non_shared = np.where(assignment != SHARED)[0]
    # Pre-sort exp_sq columns to get top-k info per token
    # sorted_idx[t] gives indices of experts sorted by exp_sq[:, t] ascending
    sorted_idx = np.argsort(exp_sq, axis=0)  # shape: (n_experts, T) - ascending order
    # top_k_sum = sum of last top_k rows
    cur_cov = exp_sq[sorted_idx[-top_k:, :], np.arange(T)].sum()

    n_swaps = 0
    rng = np.random.default_rng(rng_seed)
    order = rng.permutation(non_shared)
    expert_chs = [list(non_shared[assignment[non_shared] == e]) for e in range(n_experts)]

    for i in order:
        src_i = int(assignment[i])
        sq_i = act_sq[:, i]

        best_cov_swap = cur_cov
        best_j = -1
        best_src_j = -1

        for dst_e in range(n_experts):
            if dst_e == src_i:
                continue
            chs_j = expert_chs[dst_e]
            if not chs_j:
                continue
            if len(chs_j) <= n_candidates:
                candidates = chs_j
            else:
                idx = rng.integers(0, len(chs_j), size=n_candidates)
                candidates = [chs_j[k] for k in idx]

            for j in candidates:
                sq_j = act_sq[:, j]
                # Compute new proxy for this swap without copying full exp_sq
                # Modified experts: src_i and dst_e
                # new_src = exp_sq[src_i] - sq_i + sq_j
                # new_dst = exp_sq[dst_e] - sq_j + sq_i
                new_src = exp_sq[src_i] - sq_i + sq_j
                new_dst = exp_sq[dst_e] - sq_j + sq_i

                # For each token t, compute new top-k sum
                # Current top-k is sorted_idx[-top_k:, t]
                # We need to update for experts src_i and dst_e
                # For each token: old values of src_i and dst_e are exp_sq[src_i,t] and exp_sq[dst_e,t]
                # New values are new_src[t] and new_dst[t]
                # Quick version: compute trial proxy
                trial = exp_sq.copy()
                trial[src_i] = new_src
                trial[dst_e] = new_dst
                cov = np.sort(trial, axis=0)[-top_k:, :].sum()
                if cov > best_cov_swap:
                    best_cov_swap = cov
                    best_j = j
                    best_src_j = dst_e

        if best_j >= 0:
            sq_j = act_sq[:, best_j]
            exp_sq[src_i] = exp_sq[src_i] - sq_i + sq_j
            exp_sq[best_src_j] = exp_sq[best_src_j] - sq_j + sq_i
            # Re-sort the affected columns
            sorted_idx = np.argsort(exp_sq, axis=0)
            expert_chs[src_i].remove(int(i))
            expert_chs[src_i].append(int(best_j))
            expert_chs[best_src_j].remove(int(best_j))
            expert_chs[best_src_j].append(int(i))
            assignment[i] = best_src_j
            assignment[best_j] = src_i
            cur_cov = best_cov_swap
            n_swaps += 1

    return n_swaps, assignment, cur_cov


def _swap_sweep_nocopy(activations, assignment, n_experts, top_k, rng_seed, n_candidates):
    """Swap sweep without exp_sq.copy() - use in-place delta and check."""
    T, I = activations.shape
    act_sq = activations ** 2

    exp_sq = np.zeros((n_experts, T))
    for e in range(n_experts):
        mask = assignment == e
        if mask.any():
            exp_sq[e] = act_sq[:, mask].sum(axis=1)

    non_shared = np.where(assignment != SHARED)[0]

    # Pre-sort for fast top-k lookup
    # For each token t, sorted_exp[k, t] = k-th smallest value among experts
    sorted_exp = np.sort(exp_sq, axis=0)  # shape (n_experts, T)
    cur_cov = sorted_exp[-top_k:, :].sum()

    n_swaps = 0
    rng = np.random.default_rng(rng_seed)
    order = rng.permutation(non_shared)
    expert_chs = [list(non_shared[assignment[non_shared] == e]) for e in range(n_experts)]

    for i in order:
        src_i = int(assignment[i])
        sq_i = act_sq[:, i]

        best_cov_swap = cur_cov
        best_j = -1
        best_src_j = -1

        for dst_e in range(n_experts):
            if dst_e == src_i:
                continue
            chs_j = expert_chs[dst_e]
            if not chs_j:
                continue
            if len(chs_j) <= n_candidates:
                candidates = chs_j
            else:
                idx = rng.integers(0, len(chs_j), size=n_candidates)
                candidates = [chs_j[k] for k in idx]

            for j in candidates:
                sq_j = act_sq[:, j]
                # New values for src_i and dst_e
                new_src_i = exp_sq[src_i] - sq_i + sq_j  # shape (T,)
                new_dst_e = exp_sq[dst_e] - sq_j + sq_i  # shape (T,)

                # Compute proxy: for each token, replace exp_sq[src_i] and exp_sq[dst_e]
                # with new_src_i and new_dst_e, then find top-k sum
                # Efficient: start from sorted_exp, update the two changed rows
                # sorted_exp has columns sorted; we need to update 2 elements per column

                # Fast approach: delta-based proxy
                # Current: sorted_exp[-top_k:, :].sum() = cur_cov
                # After swap: for each token t, we're changing 2 elements
                # Since n_experts is small (8), just do: compute per-token sum of top-k

                # For each token t:
                #   old_vals = exp_sq[:, t] (8 values, two of which are src_i and dst_e)
                #   new_vals = same but with src_i -> new_src_i[t] and dst_e -> new_dst_e[t]
                #   old_top2 = sorted_exp[-2:, t].sum()
                #   new_top2 = max top 2 of new_vals

                # We can compute this by modifying sorted_exp for just 2 positions
                # But the issue is that inserting into sorted array is O(n_experts) per token

                # Alternative: since n_experts=8 and T=206, just do it directly
                # For each token: replace src_i and dst_e values, find new top-2
                # This is still O(n_experts * T) but avoids the full array copy

                # Modified exp_sq for these 2 rows only
                old_src = exp_sq[src_i].copy()
                old_dst = exp_sq[dst_e].copy()
                exp_sq[src_i] = new_src_i
                exp_sq[dst_e] = new_dst_e
                cov = np.sort(exp_sq, axis=0)[-top_k:, :].sum()
                exp_sq[src_i] = old_src
                exp_sq[dst_e] = old_dst

                if cov > best_cov_swap:
                    best_cov_swap = cov
                    best_j = j
                    best_src_j = dst_e

        if best_j >= 0:
            sq_j = act_sq[:, best_j]
            exp_sq[src_i] = exp_sq[src_i] - sq_i + sq_j
            exp_sq[best_src_j] = exp_sq[best_src_j] - sq_j + sq_i
            sorted_exp = np.sort(exp_sq, axis=0)
            expert_chs[src_i].remove(int(i))
            expert_chs[src_i].append(int(best_j))
            expert_chs[best_src_j].remove(int(best_j))
            expert_chs[best_src_j].append(int(i))
            assignment[i] = best_src_j
            assignment[best_j] = src_i
            cur_cov = best_cov_swap
            n_swaps += 1

    return n_swaps, assignment, cur_cov


d3 = np.load('examples/grouping-search/layer3.npz')
act = d3['activations'].astype(np.float64)
imp = d3['importance'].astype(np.float64)
down = d3['down'].astype(np.float64)

n_experts = 8
shared_ratio = 0.125
T, I = act.shape
n_shared = int(round(shared_ratio * I))
n_routed = I - n_shared
max_size = int(2.0 * n_routed / n_experts)

rng = np.random.default_rng(0)
sub_rng = np.random.default_rng(rng.integers(0, 2**31))
assign_base = _make_seed(act, imp, n_experts, shared_ratio, sub_rng)
proxy = 0.0
for sw in range(60):
    n_moves, assign_base, proxy = _sweep(act, assign_base, n_experts, 2, sw, max_size)
    if n_moves == 0:
        break
print(f'After standard sweep: proxy={proxy:.2f}')

# Time original vs no-copy approach
from moeforge.grouping import oracle_topk_error
import copy

def _swap_sweep_orig(activations, assignment, n_experts, top_k, rng_seed, n_candidates):
    T, I = activations.shape
    act_sq = activations ** 2
    exp_sq = np.zeros((n_experts, T))
    for e in range(n_experts):
        mask = assignment == e
        if mask.any():
            exp_sq[e] = act_sq[:, mask].sum(axis=1)
    non_shared = np.where(assignment != SHARED)[0]
    cur_cov = np.sort(exp_sq, axis=0)[-top_k:, :].sum()
    n_swaps = 0
    rng = np.random.default_rng(rng_seed)
    order = rng.permutation(non_shared)
    expert_chs = [list(non_shared[assignment[non_shared] == e]) for e in range(n_experts)]
    for i in order:
        src_i = int(assignment[i])
        sq_i = act_sq[:, i]
        best_cov_swap = cur_cov
        best_j = -1
        best_src_j = -1
        for dst_e in range(n_experts):
            if dst_e == src_i:
                continue
            chs_j = expert_chs[dst_e]
            if not chs_j:
                continue
            if len(chs_j) <= n_candidates:
                candidates = chs_j
            else:
                idx = rng.integers(0, len(chs_j), size=n_candidates)
                candidates = [chs_j[k] for k in idx]
            for j in candidates:
                sq_j = act_sq[:, j]
                trial = exp_sq.copy()
                trial[src_i] = exp_sq[src_i] - sq_i + sq_j
                trial[dst_e] = exp_sq[dst_e] - sq_j + sq_i
                cov = np.sort(trial, axis=0)[-top_k:, :].sum()
                if cov > best_cov_swap:
                    best_cov_swap = cov
                    best_j = j
                    best_src_j = dst_e
        if best_j >= 0:
            sq_j = act_sq[:, best_j]
            exp_sq[src_i] = exp_sq[src_i] - sq_i + sq_j
            exp_sq[best_src_j] = exp_sq[best_src_j] - sq_j + sq_i
            expert_chs[src_i].remove(int(i))
            expert_chs[src_i].append(int(best_j))
            expert_chs[best_src_j].remove(int(best_j))
            expert_chs[best_src_j].append(int(i))
            assignment[i] = best_src_j
            assignment[best_j] = src_i
            cur_cov = best_cov_swap
            n_swaps += 1
    return n_swaps, assignment, cur_cov

N = 3
t0 = time.time()
for k in range(N):
    assign2 = assign_base.copy()
    n_sw, assign2, proxy2 = _swap_sweep_orig(act, assign2, n_experts, 2, k * 7777, 6)
print(f'Original swap (x{N}): {(time.time()-t0)/N:.2f}s, swaps={n_sw}, proxy={proxy2:.2f}')

t0 = time.time()
for k in range(N):
    assign2 = assign_base.copy()
    n_sw, assign2, proxy2 = _swap_sweep_nocopy(act, assign2, n_experts, 2, k * 7777, 6)
print(f'No-copy swap (x{N}): {(time.time()-t0)/N:.2f}s, swaps={n_sw}, proxy={proxy2:.2f}')
