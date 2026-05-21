"""Test perturbation-based escape from local optima."""
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
    """Fast top-k coverage sweep. Returns (n_moves, assignment, proxy)."""
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


def perturb_assignment(assignment, n_experts, max_size, rng, n_perturb=20):
    """Randomly swap n_perturb channels between expert pairs."""
    non_shared = np.where(assignment != SHARED)[0]
    counts = np.bincount(assignment[non_shared], minlength=n_experts)

    for _ in range(n_perturb):
        # Pick random non-shared channel
        idx = rng.choice(non_shared)
        src = int(assignment[idx])

        # Pick random destination that isn't full
        attempts = 0
        while attempts < 10:
            dst = int(rng.integers(0, n_experts))
            if dst != src and counts[dst] < max_size and counts[src] > 1:
                assignment[idx] = dst
                counts[src] -= 1
                counts[dst] += 1
                break
            attempts += 1

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

    # Phase 1: get a converged assignment
    rng = np.random.default_rng(42)
    assign = make_seed(act, imp, n_experts, shared_ratio, rng)
    for sweep in range(60):
        n_moves, assign, proxy = sweep_fast(act, assign, n_experts, 2, sweep, max_size)
        if n_moves == 0: break
    err0 = oracle_topk_error(activations=act, down=down, assignment=assign, top_k=2)
    counts0 = np.bincount(assign[assign != SHARED], minlength=n_experts)
    print(f'  Initial converged: err={err0:.6f}, proxy={proxy:.2f}, sizes={counts0.tolist()}')

    best_proxy = proxy
    best_assign = assign.copy()

    # Phase 2: perturb + re-sweep, many times
    t0 = time.time()
    n_restarts = 0
    rng2 = np.random.default_rng(999)
    while time.time() - t0 < 20:
        # Perturb the best assignment
        trial = best_assign.copy()
        trial = perturb_assignment(trial, n_experts, max_size, rng2, n_perturb=50)

        for sweep in range(30):
            n_moves, trial, proxy = sweep_fast(act, trial, n_experts, 2, n_restarts*100+sweep, max_size)
            if n_moves == 0: break

        if proxy > best_proxy:
            best_proxy = proxy
            best_assign = trial.copy()
            err_new = oracle_topk_error(activations=act, down=down, assignment=best_assign, top_k=2)
            counts_new = np.bincount(best_assign[best_assign != SHARED], minlength=n_experts)
            print(f'  Improved at restart {n_restarts}: err={err_new:.6f}, proxy={proxy:.2f}')
        n_restarts += 1

    err = oracle_topk_error(activations=act, down=down, assignment=best_assign, top_k=2)
    counts = np.bincount(best_assign[best_assign != SHARED], minlength=n_experts)
    print(f'  After {n_restarts} perturbation restarts ({time.time()-t0:.1f}s): err={err:.6f}, sizes={counts.tolist()}')

print('\nDone.')
