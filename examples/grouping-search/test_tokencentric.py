"""Test token-centric channel assignment approaches."""
import sys, time
sys.path.insert(0, 'src')
import numpy as np
from moeforge.grouping import oracle_topk_error, SHARED

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


def make_seed_token_cluster(activations, importance, n_experts, shared_ratio, rng, max_size):
    """Token-centric seed: cluster tokens, then assign channels to token clusters.

    Approach:
    1. First, assign top-importance channels as SHARED.
    2. For remaining channels, compute their activation vector = act[:, i] (normalized).
    3. Cluster channels DIRECTLY by their token activation pattern.
       (This is what cosine k-means does!)

    Alternative token-centric:
    1. Cluster TOKENS by their activation profile act[t, :] (normalized over channels).
    2. Assign each token cluster to one of n_experts "token types".
    3. For each channel, assign it to the token-type expert for which it has highest
       mean sq-activation.
    4. Balance by moving low-energy channels from oversized to undersized.
    """
    I = importance.shape[0]
    T = activations.shape[0]
    n_shared = int(round(shared_ratio * I))
    order = np.argsort(-importance)
    assignment = np.empty(I, dtype=int)
    assignment[order[:n_shared]] = SHARED
    remaining = order[n_shared:]

    # For token clustering: use sq-activations over remaining channels
    act_rem = activations[:, remaining]  # [T, n_rem]
    # Token vectors: each token's sq-activation profile
    tok_sq = (act_rem ** 2)  # [T, n_rem]
    tok_norms = np.linalg.norm(tok_sq, axis=1, keepdims=True) + 1e-12
    tok_normed = tok_sq / tok_norms  # [T, n_rem]

    # Cluster tokens into n_experts groups
    tok_labels = _kmeans(tok_normed, n_experts, rng, iters=20)  # [T]

    # For each remaining channel, assign to the expert cluster with highest mean sq-act
    n_rem = len(remaining)
    ch_scores = np.zeros((n_rem, n_experts))  # [n_rem, E]
    for e in range(n_experts):
        tok_mask = tok_labels == e
        if tok_mask.any():
            ch_scores[:, e] = (act_rem[tok_mask, :] ** 2).mean(axis=0)

    labels = ch_scores.argmax(axis=1)  # [n_rem]

    for ch, lab in zip(remaining, labels):
        assignment[ch] = int(lab)

    # Balance repair: move lowest-energy channels from oversized experts
    ch_energy = dict(zip(remaining.tolist(), (activations[:, remaining]**2).sum(axis=0).tolist()))

    for _ in range(100000):
        counts = np.bincount(assignment[remaining], minlength=n_experts)
        if counts.max() <= max_size:
            break
        src = int(np.argmax(counts))
        src_chs = remaining[assignment[remaining] == src]
        energies = np.array([ch_energy[c] for c in src_chs])
        lowest_idx = int(np.argmin(energies))
        ch = src_chs[lowest_idx]
        dst = int(np.argmin(counts))
        assignment[ch] = dst

    return assignment


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
    print(f'\n=== {layer_name} ===')

    # Test token-centric seeds with multiple restarts
    t0 = time.time()
    best_ps = -1
    best_assign = None

    for try_idx in range(10):
        if time.time() - t0 > 22: break
        rng = np.random.default_rng(try_idx * 37 + 200)
        assign = make_seed_token_cluster(act, imp, n_experts, shared_ratio, rng, max_size)
        counts_seed = np.bincount(assign[assign != SHARED], minlength=n_experts)
        err_seed = oracle_topk_error(activations=act, down=down, assignment=assign, top_k=2)
        if try_idx < 3:
            print(f'  Token-centric seed {try_idx}: err={err_seed:.6f}, sizes={counts_seed.tolist()}')

        for sweep in range(60):
            n_moves, assign, proxy = sweep_fast(act, assign, n_experts, 2, try_idx*100+sweep, max_size)
            if n_moves == 0: break

        if proxy > best_ps:
            best_ps = proxy
            best_assign = assign.copy()

    err = oracle_topk_error(activations=act, down=down, assignment=best_assign, top_k=2)
    counts = np.bincount(best_assign[best_assign != SHARED], minlength=n_experts)
    print(f'  Token-centric (multi-restart): err={err:.6f}, sizes={counts.tolist()}, time={time.time()-t0:.1f}s')

print('\nDone.')
