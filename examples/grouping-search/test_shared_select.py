"""Test different SHARED channel selection strategies."""
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


def run_with_shared_strategy(act, down, imp, n_experts, shared_ratio, shared_strategy, n_tries=10, time_limit=22):
    I = imp.shape[0]
    n_shared = int(round(shared_ratio * I))
    n_routed = I - n_shared
    max_size = int(2.0 * n_routed / n_experts)

    # Select shared channels by strategy
    if shared_strategy == 'importance':
        # Default: top by importance
        shared_idx = np.argsort(-imp)[:n_shared]
    elif shared_strategy == 'uniform_act':
        # Channels with most uniform activation across tokens (low variance)
        ch_var = (act ** 2).var(axis=0)
        shared_idx = np.argsort(ch_var)[:n_shared]  # lowest variance
    elif shared_strategy == 'high_energy_uniform':
        # High energy AND uniform: high mean act^2, low variance/mean
        act_sq = act ** 2
        ch_mean = act_sq.mean(axis=0)
        ch_var = act_sq.var(axis=0)
        ch_cv = ch_var / (ch_mean + 1e-12)
        # Score: high mean, low cv
        score = ch_mean / (ch_cv + 1e-3)
        shared_idx = np.argsort(-score)[:n_shared]
    elif shared_strategy == 'down_coverage':
        # This requires down - test what "perfect" shared selection would be
        # Select channels whose removal from SHARED would increase error most
        # This is the oracle SHARED selection - not usable in group() but shows upper bound
        pass

    assignment = np.full(I, -999, dtype=int)  # -999 = unassigned
    assignment[shared_idx] = SHARED

    routed_idx = np.where(assignment != SHARED)[0]
    n_routed = len(routed_idx)

    # Use cosine k-means for routed channels
    best_ps = -1
    best_assign = None
    t0 = time.time()
    for try_idx in range(n_tries):
        if time.time() - t0 > time_limit: break
        rng = np.random.default_rng(try_idx * 17 + 42)
        vecs = act[:, routed_idx].T
        vecs = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12)
        labels = _kmeans(vecs, n_experts, rng)
        assign = assignment.copy()
        for ch, lab in zip(routed_idx, labels):
            assign[ch] = int(lab)

        for sweep in range(60):
            n_moves, assign, proxy = sweep_fast(act, assign, n_experts, 2, try_idx*100+sweep, max_size)
            if n_moves == 0: break
        if proxy > best_ps:
            best_ps = proxy
            best_assign = assign.copy()

    err = oracle_topk_error(activations=act, down=down, assignment=best_assign, top_k=2)
    counts = np.bincount(best_assign[best_assign != SHARED], minlength=n_experts)
    return err, counts, time.time()-t0


d3 = np.load('examples/grouping-search/layer3.npz')
d9 = np.load('examples/grouping-search/layer9.npz')
layers = [('layer3', d3), ('layer9', d9)]

n_experts = 8
shared_ratio = 0.125

for layer_name, data in layers:
    act = data['activations'].astype(np.float64)
    imp = data['importance'].astype(np.float64)
    down = data['down'].astype(np.float64)
    print(f'\n=== {layer_name} ===')

    for strat in ['importance', 'uniform_act', 'high_energy_uniform']:
        err, counts, dt = run_with_shared_strategy(act, down, imp, n_experts, shared_ratio, strat, n_tries=10, time_limit=22)
        print(f'  {strat}: err={err:.6f}, sizes={counts.tolist()}, t={dt:.1f}s')

print('\nDone.')
