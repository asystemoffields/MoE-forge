"""Test early stopping strategy for restarts."""
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


def make_seed(activations, importance, n_experts, shared_ratio, rng):
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


def proxy_score(act_sq, exp_sq, top_k=2):
    return float(np.sort(exp_sq, axis=0)[-top_k:, :].sum())


def compute_initial_proxy(activations, assignment, n_experts):
    T, I = activations.shape
    act_sq = activations ** 2
    exp_sq = np.zeros((n_experts, T))
    for e in range(n_experts):
        mask = assignment == e
        if mask.any():
            exp_sq[e] = act_sq[:, mask].sum(axis=1)
    return proxy_score(act_sq, exp_sq)


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

    # Collect data: seed proxy vs. final proxy vs. oracle error
    print('  seed_proxy, after1sweep_proxy, final_proxy, oracle_err, time')
    seed_proxies = []
    final_proxies = []
    errs = []

    for try_idx in range(40):
        rng = np.random.default_rng(try_idx * 17 + 2000)
        t0 = time.time()
        assign = make_seed(act, imp, n_experts, shared_ratio, rng)
        sp = compute_initial_proxy(act, assign, n_experts)

        n_moves, assign, proxy1 = sweep_fast(act, assign, n_experts, 2, try_idx*100, max_size)
        final_proxy = proxy1
        for sweep in range(1, 60):
            n_moves, assign, proxy = sweep_fast(act, assign, n_experts, 2, try_idx*100+sweep, max_size)
            final_proxy = proxy
            if n_moves == 0: break
        dt = time.time() - t0

        err = oracle_topk_error(activations=act, down=down, assignment=assign, top_k=2)
        seed_proxies.append(sp)
        final_proxies.append(final_proxy)
        errs.append(err)
        print(f'  {sp:.2f}, {proxy1:.2f}, {final_proxy:.2f}, {err:.6f}, {dt:.2f}s')

    print(f'\n  Correlation(seed_proxy, final_proxy): {np.corrcoef(seed_proxies, final_proxies)[0,1]:.4f}')
    print(f'  Correlation(seed_proxy, oracle_err): {np.corrcoef(seed_proxies, errs)[0,1]:.4f}')
    print(f'  Correlation(final_proxy, oracle_err): {np.corrcoef(final_proxies, errs)[0,1]:.4f}')

    # What threshold on seed_proxy would select the top 50% best restarts?
    best_half_idx = np.argsort(final_proxies)[-len(final_proxies)//2:]
    threshold_seed_proxy = np.sort(np.array(seed_proxies)[best_half_idx])[0]
    print(f'  Seed proxy threshold for top-50% final proxies: {threshold_seed_proxy:.2f}')
    print(f'  (median seed proxy: {np.median(seed_proxies):.2f})')

print('\nDone.')
