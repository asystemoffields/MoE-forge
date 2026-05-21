"""Profile gen1_factor performance: how many restarts in 22s? What's the distribution?"""
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
act = d3['activations'].astype(np.float64)
imp = d3['importance'].astype(np.float64)
down = d3['down'].astype(np.float64)
I = imp.shape[0]
n_shared = int(round(0.125 * I))
n_routed = I - n_shared
max_size = int(2.0 * n_routed / 8)
print(f'layer3: I={I}, n_routed={n_routed}, max_size={max_size}')

proxies = []
errs = []
times = []
n_sweeps_list = []

for try_idx in range(30):
    t0 = time.time()
    rng = np.random.default_rng(try_idx * 17 + 1000)
    assign = make_seed(act, imp, 8, 0.125, rng)
    t_seed = time.time() - t0

    t1 = time.time()
    n_sw = 0
    for sweep in range(60):
        n_moves, assign, proxy = sweep_fast(act, assign, 8, 2, try_idx*100+sweep, max_size)
        n_sw += 1
        if n_moves == 0: break
    t_sweep = time.time() - t1

    err = oracle_topk_error(activations=act, down=down, assignment=assign, top_k=2)
    proxies.append(proxy)
    errs.append(err)
    times.append(t_seed + t_sweep)
    n_sweeps_list.append(n_sw)
    print(f'  Restart {try_idx}: proxy={proxy:.2f}, err={err:.6f}, t={t_seed+t_sweep:.2f}s, n_sweeps={n_sw}')

print(f'\nSummary:')
print(f'  Total time: {sum(times):.1f}s')
print(f'  Best proxy: {max(proxies):.2f} (at restart {np.argmax(proxies)})')
print(f'  Best err: {min(errs):.6f} (at restart {np.argmin(errs)})')
print(f'  Restarts achievable in 22s: {sum(1 for t in np.cumsum(times) if t < 22)}')
print(f'  Avg time per restart: {np.mean(times):.2f}s')
print(f'  Proxy distribution: min={min(proxies):.2f}, max={max(proxies):.2f}, median={np.median(proxies):.2f}')
print(f'  Best proxy selects err: {errs[np.argmax(proxies)]:.6f}')
