"""Test normalized proxy sweep variants."""
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


def sweep_normed(activations, assignment, n_experts, top_k, rng_seed, max_size):
    """Sweep maximizing top-k of *normalized* expert sq-activation fractions.

    Objective: for each token t, compute frac[e,t] = exp_sq[e,t] / sum_e exp_sq[e,t],
    then maximize sum_t top_k_sum(frac[:, t]).
    This is equivalent to maximizing top-k share of energy in each expert.
    """
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

    total_sq = exp_sq.sum(axis=0)  # [T] total sq-act per token

    def score(esq, tot):
        # Normalized fractions, then top-k sum
        frac = esq / (tot[np.newaxis, :] + 1e-12)  # [E, T]
        return np.sort(frac, axis=0)[-top_k:, :].sum()

    cur_cov = score(exp_sq, total_sq)

    for i in order:
        src = int(assignment[i])
        sq_i = act_sq[:, i]
        new_sq_src = exp_sq[src] - sq_i

        # Total doesn't change when moving channels (all channels remain in some expert)

        best_cov = cur_cov
        best_dst = src

        for dst in range(n_experts):
            if dst == src: continue
            if exp_sz[dst] >= max_size: continue
            if exp_sz[src] <= 1: continue
            trial = exp_sq.copy()
            trial[src] = new_sq_src
            trial[dst] = exp_sq[dst] + sq_i
            cov = score(trial, total_sq)
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


def sweep_rank_topk(activations, assignment, n_experts, top_k, rng_seed, max_size):
    """Count-based: for each token, is this expert in top-k? Maximize sum of top-k indicator."""
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

    def score(esq):
        # For each token, which experts are top-k? Sum of top-k exp_sq values
        return np.sort(esq, axis=0)[-top_k:, :].sum()

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

    return n_moves, assignment, cur_cov


# Test: what does the seed look like in terms of oracle error?
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

    # Test seed quality before refinement
    rng = np.random.default_rng(42)
    assign0 = make_seed(act, imp, n_experts, shared_ratio, rng)
    err0 = oracle_topk_error(activations=act, down=down, assignment=assign0, top_k=2)
    counts0 = np.bincount(assign0[assign0 != SHARED], minlength=n_experts)
    print(f'  Seed: err={err0:.6f}, sizes={counts0.tolist()}')

    # Test normed sweep
    rng = np.random.default_rng(42)
    assign = make_seed(act, imp, n_experts, shared_ratio, rng)
    for sweep in range(60):
        n_moves, assign, proxy = sweep_normed(act, assign, n_experts, 2, sweep, max_size)
        if n_moves == 0: break
    err = oracle_topk_error(activations=act, down=down, assignment=assign, top_k=2)
    counts = np.bincount(assign[assign != SHARED], minlength=n_experts)
    print(f'  normed sweep: err={err:.6f}, sizes={counts.tolist()}')

    # Multi-restart normed sweep
    t0 = time.time()
    best_ps = -1
    best_assign = None
    for try_idx in range(20):
        if time.time() - t0 > 22: break
        rng = np.random.default_rng(try_idx * 17 + 99)
        assign = make_seed(act, imp, n_experts, shared_ratio, rng)
        for sweep in range(60):
            n_moves, assign, proxy = sweep_normed(act, assign, n_experts, 2, sweep*3+try_idx, max_size)
            if n_moves == 0: break
        if proxy > best_ps:
            best_ps = proxy
            best_assign = assign.copy()
    err = oracle_topk_error(activations=act, down=down, assignment=best_assign, top_k=2)
    counts = np.bincount(best_assign[best_assign != SHARED], minlength=n_experts)
    print(f'  normed sweep (multi): err={err:.6f}, sizes={counts.tolist()}, time={time.time()-t0:.1f}s')

print('\nDone.')
