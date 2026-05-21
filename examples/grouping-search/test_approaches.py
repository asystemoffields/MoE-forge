"""Quick test of several different sweep objectives to find lower oracle error."""
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


def sweep_topk(activations, assignment, n_experts, top_k, rng_seed, max_size):
    """Standard top-k proxy sweep."""
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


def sweep_top4(activations, assignment, n_experts, rng_seed, max_size):
    """Sweep maximizing top-4 coverage (then test with top-2 oracle)."""
    top_k = 4
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
    return n_moves, assignment


def sweep_logsumexp(activations, assignment, n_experts, rng_seed, max_size, temp=1.0):
    """Sweep maximizing sum of log(1 + exp(expert_sq)) -- softer top-k."""
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
    # Normalize for numerical stability
    scale = exp_sq.max() + 1e-12
    def score(esq):
        return np.log1p(esq / scale).sum()
    cur_cov = score(exp_sq)
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
            cov = score(trial)
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
    return n_moves, assignment


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
    print(f'\n=== {layer_name} === (I={I}, n_routed={n_routed}, max_size={max_size})')

    # Approach 1: standard top-2 sweep (gen1_factor baseline)
    rng = np.random.default_rng(42)
    assign = make_seed(act, imp, n_experts, shared_ratio, rng)
    for sweep in range(60):
        n_moves, assign, proxy = sweep_topk(act, assign, n_experts, 2, sweep, max_size)
        if n_moves == 0: break
    err = oracle_topk_error(activations=act, down=down, assignment=assign, top_k=2)
    counts = np.bincount(assign[assign != SHARED], minlength=n_experts)
    print(f'  top-2 sweep: err={err:.6f}, sizes={counts.tolist()}')

    # Approach 2: top-4 sweep (optimize top-4 coverage for better balance)
    rng = np.random.default_rng(42)
    assign = make_seed(act, imp, n_experts, shared_ratio, rng)
    for sweep in range(60):
        n_moves, assign = sweep_top4(act, assign, n_experts, sweep, max_size)
        if n_moves == 0: break
    err = oracle_topk_error(activations=act, down=down, assignment=assign, top_k=2)
    counts = np.bincount(assign[assign != SHARED], minlength=n_experts)
    print(f'  top-4 sweep: err={err:.6f}, sizes={counts.tolist()}')

    # Approach 3: logsumexp sweep (smooth coverage, avoids winner-takes-all)
    rng = np.random.default_rng(42)
    assign = make_seed(act, imp, n_experts, shared_ratio, rng)
    for sweep in range(60):
        n_moves, assign = sweep_logsumexp(act, assign, n_experts, sweep, max_size)
        if n_moves == 0: break
    err = oracle_topk_error(activations=act, down=down, assignment=assign, top_k=2)
    counts = np.bincount(assign[assign != SHARED], minlength=n_experts)
    print(f'  logsumexp sweep: err={err:.6f}, sizes={counts.tolist()}')

    # Approach 4: top-2 sweep with STRICT balance (max_size = ideal*1.5)
    strict_max = int(1.5 * n_routed / n_experts)
    rng = np.random.default_rng(42)
    assign = make_seed(act, imp, n_experts, shared_ratio, rng)
    for sweep in range(60):
        n_moves, assign, proxy = sweep_topk(act, assign, n_experts, 2, sweep, strict_max)
        if n_moves == 0: break
    err = oracle_topk_error(activations=act, down=down, assignment=assign, top_k=2)
    counts = np.bincount(assign[assign != SHARED], minlength=n_experts)
    print(f'  top-2 strict(1.5x) sweep: err={err:.6f}, sizes={counts.tolist()}')

print('\nDone.')
