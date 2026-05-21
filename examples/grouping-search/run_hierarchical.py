"""Hierarchical merging: k-means with 2x experts, merge pairs to get 8 balanced groups."""
import sys, time
sys.path.insert(0, 'src')
import numpy as np
from moeforge.grouping import oracle_topk_error, SHARED

SHARED = -1

def _kmeans(points, k, rng, iters=30):
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
    return labels, centers


def sweep_fast(activations, assignment, n_experts, top_k, rng_seed, max_size):
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
            if dst == src: continue
            if exp_sz[dst] >= max_size: continue
            if exp_sz[src] <= 1: continue
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


def make_hierarchical_seed(activations, importance, n_experts, shared_ratio, rng, max_size, k_init=None):
    """
    1. Run k-means with k_init clusters (default 2*n_experts).
    2. Compute proxy for merging each pair of clusters.
    3. Greedily merge pairs that maximize proxy, subject to balance.
    Returns assignment with n_experts clusters.
    """
    if k_init is None:
        k_init = 2 * n_experts

    I = importance.shape[0]
    T = activations.shape[0]
    n_shared = int(round(shared_ratio * I))
    order = np.argsort(-importance)
    assignment = np.empty(I, dtype=int)
    assignment[order[:n_shared]] = SHARED
    remaining = order[n_shared:]

    vecs = activations[:, remaining].T
    vecs = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12)
    labels, centers = _kmeans(vecs, k_init, rng)

    # Map to remaining indices
    cluster_assign = np.zeros(len(remaining), dtype=int)
    for i, lab in enumerate(labels):
        cluster_assign[i] = lab

    # Act_sq for proxy computation
    act_sq = activations ** 2

    # Compute per-cluster sq-activation sums
    cluster_exp_sq = np.zeros((k_init, T))
    cluster_size = np.zeros(k_init, dtype=int)
    for c in range(k_init):
        mask = cluster_assign == c
        cluster_size[c] = mask.sum()
        if mask.any():
            ch_indices = remaining[mask]
            cluster_exp_sq[c] = act_sq[:, ch_indices].sum(axis=1)

    # Current mapping: each of k_init clusters maps to one of n_experts experts
    # Start with: cluster c -> expert c % n_experts (round-robin)
    # We'll merge clusters by proxy greedy

    # Actually, let's do hierarchical merging:
    # Start with k_init clusters. At each step, merge the pair of clusters
    # whose merging results in the highest proxy improvement, subject to
    # merged cluster size <= max_size.

    # Current state
    alive = list(range(k_init))
    merged_into = {c: c for c in alive}  # cluster c is merged into which group

    def current_proxy(exp_sq_list):
        exp_sq_mat = np.stack(exp_sq_list)  # [k, T]
        return float(np.sort(exp_sq_mat, axis=0)[-2:, :].sum())

    while len(alive) > n_experts:
        best_gain = -np.inf
        best_pair = None

        for i_idx, c1 in enumerate(alive):
            for c2 in alive[i_idx+1:]:
                merged_size = cluster_size[c1] + cluster_size[c2]
                if merged_size > max_size:
                    continue

                # Try merging c1 and c2
                merged_sq = cluster_exp_sq[c1] + cluster_exp_sq[c2]

                # Compute proxy with this merge
                test_expsq = [cluster_exp_sq[c] for c in alive if c != c1 and c != c2]
                test_expsq.append(merged_sq)
                new_proxy = current_proxy(test_expsq)
                base_proxy = current_proxy([cluster_exp_sq[c] for c in alive])
                gain = new_proxy - base_proxy

                if gain > best_gain:
                    best_gain = gain
                    best_pair = (c1, c2, merged_sq, merged_size)

        if best_pair is None:
            # No valid merge possible - just merge smallest pair
            # Sort by size, merge two smallest
            sorted_by_size = sorted(alive, key=lambda c: cluster_size[c])
            c1, c2 = sorted_by_size[0], sorted_by_size[1]
            merged_sq = cluster_exp_sq[c1] + cluster_exp_sq[c2]
            merged_size = cluster_size[c1] + cluster_size[c2]
            best_pair = (c1, c2, merged_sq, merged_size)

        c1, c2, merged_sq, merged_size = best_pair
        # Merge c2 into c1
        cluster_exp_sq[c1] = merged_sq
        cluster_size[c1] = merged_size
        # Update all channels in c2 to be in c1
        for i, lab in enumerate(cluster_assign):
            if lab == c2:
                cluster_assign[i] = c1
        alive.remove(c2)

    # Remap alive clusters to 0..n_experts-1
    remap = {c: i for i, c in enumerate(alive)}
    for ch_idx, lab in enumerate(cluster_assign):
        assignment[remaining[ch_idx]] = remap[lab]

    return assignment


out = open('examples/grouping-search/hierarchical_results.txt', 'w')

def log(s):
    out.write(s + '\n')
    out.flush()


d3 = np.load('examples/grouping-search/layer3.npz')
d9 = np.load('examples/grouping-search/layer9.npz')
layers = [('layer3', d3), ('layer9', d9)]

n_experts = 8
shared_ratio = 0.125

for layer_name, data in layers:
    act = data['activations'].astype(np.float64)
    imp = data['importance'].astype(np.float64)
    down = data['down'].astype(np.float64)
    I = imp.shape[0]
    n_routed = I - int(round(shared_ratio * I))
    max_size = int(2.0 * n_routed / n_experts)
    log(f'\n=== {layer_name} === (max_size={max_size})')

    t_total = time.time()
    best_ps = -1
    best_assign = None

    for try_idx in range(10):
        if time.time() - t_total > 22: break
        rng = np.random.default_rng(try_idx * 17 + 500)

        t0 = time.time()
        assign = make_hierarchical_seed(act, imp, n_experts, shared_ratio, rng, max_size, k_init=16)
        t_seed = time.time() - t0

        counts = np.bincount(assign[assign != SHARED], minlength=n_experts)
        err_seed = oracle_topk_error(activations=act, down=down, assignment=assign, top_k=2)
        log(f'  Hierarchical seed {try_idx}: err={err_seed:.6f}, sizes={counts.tolist()}, seed_time={t_seed:.2f}s')

        for sweep in range(60):
            n_moves, assign, proxy = sweep_fast(act, assign, n_experts, 2, try_idx*100+sweep, max_size)
            if n_moves == 0: break
        if proxy > best_ps:
            best_ps = proxy
            best_assign = assign.copy()

    err = oracle_topk_error(activations=act, down=down, assignment=best_assign, top_k=2)
    counts = np.bincount(best_assign[best_assign != SHARED], minlength=n_experts)
    log(f'  Best hierarchical (10 tries): err={err:.6f}, sizes={counts.tolist()}, t={time.time()-t_total:.1f}s')

log('\nDone.')
out.close()
