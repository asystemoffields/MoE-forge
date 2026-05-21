"""Test importance-weighted proxy for channel grouping.

Key hypothesis: importance[i] ≈ ||down[:, i]|| (down projection column norm).
Then expert contribution proxy becomes:
  proxy[e, t] = Σ_{i∈e} act[t,i]² * imp[i]²

This should better approximate the actual oracle contribution norm.
"""
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


def make_seed_imp_weighted(activations, importance, n_experts, shared_ratio, rng):
    """K-means on importance-weighted sq-activation vectors."""
    I = importance.shape[0]
    n_shared = int(round(shared_ratio * I))
    order = np.argsort(-importance)
    assignment = np.empty(I, dtype=int)
    assignment[order[:n_shared]] = SHARED
    remaining = order[n_shared:]

    # Channel embedding: act[:, i] * imp[i] (amplitude scaled by importance)
    imp_rem = importance[remaining]  # [n_rem]
    vecs = activations[:, remaining] * imp_rem[np.newaxis, :]  # [T, n_rem]
    vecs = vecs.T  # [n_rem, T]
    vecs = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12)
    labels = _kmeans(vecs, n_experts, rng)
    for ch, lab in zip(remaining, labels):
        assignment[ch] = int(lab)
    return assignment


def sweep_imp_weighted(activations, importance, assignment, n_experts, top_k, rng_seed, max_size):
    """Sweep using importance-weighted sq-activation proxy."""
    T, I = activations.shape
    imp_sq = importance ** 2  # [I]
    # weighted sq: act[t,i]^2 * imp[i]^2
    act_sq_w = activations ** 2 * imp_sq[np.newaxis, :]  # [T, I]

    exp_sq = np.zeros((n_experts, T))
    exp_sz = np.zeros(n_experts, dtype=int)
    for e in range(n_experts):
        mask = assignment == e
        exp_sz[e] = int(mask.sum())
        if mask.any():
            exp_sq[e] = act_sq_w[:, mask].sum(axis=1)

    non_shared = np.where(assignment != SHARED)[0]
    order = np.random.default_rng(rng_seed).permutation(non_shared)
    n_moves = 0
    cur_cov = np.sort(exp_sq, axis=0)[-top_k:, :].sum()

    for i in order:
        src = int(assignment[i])
        sq_i = act_sq_w[:, i]
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

    # Check: does importance correlate with down column norms?
    down_norms = np.linalg.norm(down, axis=0)  # [I]
    corr = np.corrcoef(imp, down_norms)[0, 1]
    print(f'  Correlation imp vs ||down[:,i]||: {corr:.4f}')

    # Test importance-weighted sweep (cosine k-means seed)
    t0 = time.time()
    best_ps = -1
    best_assign = None
    for try_idx in range(20):
        if time.time() - t0 > 22: break
        rng = np.random.default_rng(try_idx * 17 + 5)
        # Try both seed types
        if try_idx % 2 == 0:
            assign = make_seed(act, imp, n_experts, shared_ratio, rng)
        else:
            assign = make_seed_imp_weighted(act, imp, n_experts, shared_ratio, rng)

        for sweep in range(60):
            n_moves, assign, proxy = sweep_imp_weighted(act, imp, assign, n_experts, 2, try_idx*100+sweep, max_size)
            if n_moves == 0: break

        if proxy > best_ps:
            best_ps = proxy
            best_assign = assign.copy()

    err = oracle_topk_error(activations=act, down=down, assignment=best_assign, top_k=2)
    counts = np.bincount(best_assign[best_assign != SHARED], minlength=n_experts)
    print(f'  Imp-weighted proxy (multi): err={err:.6f}, sizes={counts.tolist()}, time={time.time()-t0:.1f}s')

    # Compare with standard proxy
    t0 = time.time()
    best_ps = -1
    best_assign = None
    for try_idx in range(20):
        if time.time() - t0 > 22: break
        rng = np.random.default_rng(try_idx * 17 + 5)
        assign = make_seed(act, imp, n_experts, shared_ratio, rng)
        for sweep in range(60):
            act_sq = act ** 2
            exp_sq = np.zeros((n_experts, act.shape[0]))
            exp_sz = np.zeros(n_experts, dtype=int)
            for e in range(n_experts):
                mask = assign == e
                exp_sz[e] = int(mask.sum())
                if mask.any():
                    exp_sq[e] = act_sq[:, mask].sum(axis=1)
            proxy_val = np.sort(exp_sq, axis=0)[-2:, :].sum()
            # Use sweep_imp_weighted but with unweighted proxy (standard)
            n_moves, assign, proxy = sweep_imp_weighted(act, np.ones(I), assign, n_experts, 2, try_idx*100+sweep, max_size)
            if n_moves == 0: break

        if proxy > best_ps:
            best_ps = proxy
            best_assign = assign.copy()

    err = oracle_topk_error(activations=act, down=down, assignment=best_assign, top_k=2)
    counts = np.bincount(best_assign[best_assign != SHARED], minlength=n_experts)
    print(f'  Standard proxy (multi): err={err:.6f}, sizes={counts.tolist()}, time={time.time()-t0:.1f}s')

print('\nDone.')
