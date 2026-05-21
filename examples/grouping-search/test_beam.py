"""Test beam-search restart strategy: early-abandon low-quality restarts."""
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


def group_beam(activations, importance, n_experts, shared_ratio, rng, top_k=2, time_limit=22.0,
               probe_sweeps=3, abandon_threshold_ratio=0.90):
    """Early-abandon restarts that show low proxy after probe_sweeps sweeps."""
    import time as _time
    T, I = activations.shape
    n_shared = int(round(shared_ratio * I))
    n_routed = I - n_shared
    max_size = int(2.0 * n_routed / n_experts)

    best_assign = None
    best_proxy = -1.0
    t_start = _time.time()
    trial = 0
    n_probed = 0
    n_completed = 0

    while _time.time() - t_start < time_limit:
        sub_rng = np.random.default_rng(rng.integers(0, 2**31) + trial)
        assignment = make_seed(activations, importance, n_experts, shared_ratio, sub_rng)

        # Probe phase: run probe_sweeps sweeps
        proxy = None
        for sweep in range(probe_sweeps):
            n_moves, assignment, proxy = sweep_fast(
                activations, assignment, n_experts, top_k,
                rng_seed=trial * 100 + sweep, max_size=max_size)
            if n_moves == 0:
                break

        n_probed += 1

        # Check if worth continuing
        if best_proxy > 0 and proxy < best_proxy * abandon_threshold_ratio:
            trial += 1
            continue  # Abandon this restart

        # Continue to convergence
        for sweep in range(probe_sweeps, 60):
            n_moves, assignment, proxy = sweep_fast(
                activations, assignment, n_experts, top_k,
                rng_seed=trial * 100 + sweep, max_size=max_size)
            if n_moves == 0:
                break

        n_completed += 1

        if proxy > best_proxy:
            best_proxy = proxy
            best_assign = assignment.copy()

        trial += 1

    return best_assign, best_proxy, n_probed, n_completed


d3 = np.load('examples/grouping-search/layer3.npz')
d9 = np.load('examples/grouping-search/layer9.npz')
layers = [('layer3', d3), ('layer9', d9)]

n_experts = 8
shared_ratio = 0.125

print('Testing beam search vs standard multi-restart:')
for layer_name, data in layers:
    act = data['activations'].astype(np.float64)
    imp = data['importance'].astype(np.float64)
    down = data['down'].astype(np.float64)
    I = imp.shape[0]
    n_routed = I - int(round(shared_ratio * I))
    max_size = int(2.0 * n_routed / n_experts)
    print(f'\n=== {layer_name} ===')

    rng = np.random.default_rng(0)

    for abandon_thresh in [0.85, 0.90, 0.95]:
        rng2 = np.random.default_rng(0)
        t0 = time.time()
        best_assign, best_proxy, n_probed, n_completed = group_beam(
            act, imp, n_experts, shared_ratio, rng2,
            time_limit=22.0, probe_sweeps=3, abandon_threshold_ratio=abandon_thresh)
        dt = time.time() - t0
        err = oracle_topk_error(activations=act, down=down, assignment=best_assign, top_k=2)
        counts = np.bincount(best_assign[best_assign != SHARED], minlength=n_experts)
        print(f'  beam(thresh={abandon_thresh}): err={err:.6f}, proxy={best_proxy:.2f}, '
              f'probed={n_probed}, completed={n_completed}, t={dt:.1f}s')

    # Standard multi-restart (gen1_factor baseline)
    rng3 = np.random.default_rng(0)
    t0 = time.time()
    best_ps = -1
    best_a = None
    trial = 0
    while time.time() - t0 < 22:
        sub = np.random.default_rng(rng3.integers(0, 2**31) + trial)
        assign = make_seed(act, imp, n_experts, shared_ratio, sub)
        for sw in range(60):
            n_moves, assign, proxy = sweep_fast(act, assign, n_experts, 2, trial*100+sw, max_size)
            if n_moves == 0: break
        if proxy > best_ps:
            best_ps = proxy
            best_a = assign.copy()
        trial += 1
    err = oracle_topk_error(activations=act, down=down, assignment=best_a, top_k=2)
    counts = np.bincount(best_a[best_a != SHARED], minlength=n_experts)
    print(f'  standard (no abandon): err={err:.6f}, proxy={best_ps:.2f}, completed={trial}, t={time.time()-t0:.1f}s')

print('\nDone.')
