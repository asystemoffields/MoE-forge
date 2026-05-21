"""Test swap sweep: try swapping pairs of channels between experts."""
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


def sweep_standard(activations, assignment, n_experts, top_k, rng_seed, max_size):
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


def sweep_swap(activations, assignment, n_experts, top_k, rng_seed, max_size, n_pairs=200):
    """Sweep with pairwise swaps between experts.

    For each pair of experts (A, B), try swapping the most 'mismatched' channels:
    channels from A that would prefer B and vice versa.
    Limit to n_pairs pair evaluations to keep time reasonable.
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
    cur_cov = np.sort(exp_sq, axis=0)[-top_k:, :].sum()
    n_swaps = 0
    rng = np.random.default_rng(rng_seed)

    # Try random pairs of channels from different experts
    perm = rng.permutation(non_shared)
    n_iter = min(len(perm) // 2, n_pairs)

    for k in range(n_iter):
        i = perm[k * 2]
        j = perm[k * 2 + 1]
        src_i = int(assignment[i])
        src_j = int(assignment[j])

        if src_i == src_j:
            continue  # Same expert, no swap

        sq_i = act_sq[:, i]
        sq_j = act_sq[:, j]

        # Current coverage
        # After swap: i goes to src_j, j goes to src_i
        trial = exp_sq.copy()
        trial[src_i] = exp_sq[src_i] - sq_i + sq_j
        trial[src_j] = exp_sq[src_j] - sq_j + sq_i
        cov = np.sort(trial, axis=0)[-top_k:, :].sum()

        if cov > cur_cov:
            exp_sq = trial
            assignment[i] = src_j
            assignment[j] = src_i
            cur_cov = cov
            n_swaps += 1

    return n_swaps, assignment, cur_cov


out = open('examples/grouping-search/swapsweep_results.txt', 'w')

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
    log(f'\n=== {layer_name} ===')

    # Strategy: standard sweep until convergence, then swap sweep, repeat
    rng = np.random.default_rng(0)
    t0 = time.time()
    best_ps = -1
    best_assign = None

    for try_idx in range(20):
        if time.time() - t0 > 22: break
        sub = np.random.default_rng(rng.integers(0, 2**31) + try_idx)
        assign = make_seed(act, imp, n_experts, shared_ratio, sub)

        for phase in range(5):  # Alternate standard and swap sweeps
            # Standard sweeps until convergence
            for sw in range(15):
                n_moves, assign, proxy = sweep_standard(act, assign, n_experts, 2, try_idx*100+phase*15+sw, max_size)
                if n_moves == 0: break

            # Swap sweep
            n_swaps, assign, proxy_new = sweep_swap(act, assign, n_experts, 2, try_idx*1000+phase, max_size, n_pairs=500)
            if n_swaps == 0:
                break  # No improvement from swaps either

            proxy = max(proxy, proxy_new)

        if proxy > best_ps:
            best_ps = proxy
            best_assign = assign.copy()

    err = oracle_topk_error(activations=act, down=down, assignment=best_assign, top_k=2)
    counts = np.bincount(best_assign[best_assign != SHARED], minlength=n_experts)
    log(f'  swap+standard: err={err:.6f}, proxy={best_ps:.2f}, t={time.time()-t0:.1f}s')
    log(f'  sizes={counts.tolist()}')

log('\nDone.')
out.close()
