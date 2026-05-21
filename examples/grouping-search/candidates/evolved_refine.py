"""gen6_perturb: Cosine k-means + balanced sweep + swap sweep + perturbation restarts.

Extends gen4_swap by adding a perturbation-restart strategy: alternating between
fresh cosine k-means seeds and perturbations of the best-found assignment.
Perturbation performs balance-preserving random pairwise channel swaps (20% of
routed channels), escaping the current local optimum while staying close enough
that re-convergence finds a nearby (potentially better) local optimum.

Per restart:
1. Either fresh cosine k-means seed (even trials) OR perturb best assignment (odd trials).
2. Standard balanced greedy sweep until convergence.
3. Swap sweep cycles with n_candidates=6 until convergence.
4. Keep best by proxy score.
"""

import numpy as np

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


def _swap_sweep(activations, assignment, n_experts, top_k, rng_seed, n_candidates):
    """Pairwise swap sweep: for each channel, try swapping with sampled channels
    from other experts. Swaps preserve expert sizes."""
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


def _perturb(assignment, rng, frac=0.2):
    """Balance-preserving perturbation: randomly swap frac of routed channels pairwise."""
    non_shared = np.where(assignment != SHARED)[0]
    n_shuffle = max(2, int(frac * len(non_shared)))
    n_shuffle = n_shuffle - (n_shuffle % 2)  # ensure even number for pairs
    idxs = rng.choice(len(non_shared), n_shuffle, replace=False)
    half = n_shuffle // 2
    for a, b in zip(idxs[:half], idxs[half:]):
        ch_a = non_shared[a]
        ch_b = non_shared[b]
        assignment[ch_a], assignment[ch_b] = assignment[ch_b], assignment[ch_a]
    return assignment


def group(ctx, n_experts, shared_ratio, rng, top_k=2, time_limit=30.0):
    """Partition channels using cosine k-means + sweep + swap sweep + perturbation restarts."""
    import time as _time

    activations = ctx["activations"]
    importance = ctx["importance"]
    T, I = activations.shape

    n_shared = int(round(shared_ratio * I))
    n_routed = I - n_shared
    max_size = int(2.0 * n_routed / n_experts)

    best_assign = None
    best_proxy = -1.0
    t_start = _time.time()
    trial = 0

    while _time.time() - t_start < time_limit:
        sub_rng = np.random.default_rng(rng.integers(0, 2**31) + trial)

        # Alternate: fresh seed on even trials, perturb best on odd trials
        if best_assign is not None and trial % 2 == 1:
            assignment = best_assign.copy()
            assignment = _perturb(assignment, sub_rng, frac=0.2)
        else:
            assignment = _make_seed(activations, importance, n_experts, shared_ratio, sub_rng)

        # Phase 1: standard sweep until convergence
        proxy = 0.0
        for sweep in range(60):
            n_moves, assignment, proxy = _sweep(
                activations, assignment, n_experts, top_k,
                rng_seed=trial * 100 + sweep, max_size=max_size)
            if n_moves == 0:
                break

        # Phase 2+: swap sweep cycles
        for swap_phase in range(10):
            n_swaps, assignment, proxy_new = _swap_sweep(
                activations, assignment, n_experts, top_k,
                rng_seed=trial * 1000 + swap_phase, n_candidates=6)
            if proxy_new > proxy:
                proxy = proxy_new

            if n_swaps == 0:
                break

            # Re-run standard sweep after swaps
            for sweep in range(30):
                n_moves, assignment, proxy2 = _sweep(
                    activations, assignment, n_experts, top_k,
                    rng_seed=trial * 10000 + swap_phase * 30 + sweep,
                    max_size=max_size)
                if proxy2 > proxy:
                    proxy = proxy2
                if n_moves == 0:
                    break

        if proxy > best_proxy:
            best_proxy = proxy
            best_assign = assignment.copy()

        trial += 1

    return best_assign
