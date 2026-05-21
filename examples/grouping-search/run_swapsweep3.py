"""Optimized swap+standard sweep within 22s time budget."""
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


def sweep_swap_fast(activations, assignment, n_experts, top_k, rng_seed, max_size, n_candidates=3):
    """Fast swap sweep: for each channel, try swapping with n_candidates random channels
    from each other expert."""
    T, I = activations.shape
    act_sq = activations ** 2

    exp_sq = np.zeros((n_experts, T))
    for e in range(n_experts):
        mask = assignment == e
        if mask.any():
            exp_sq[e] = act_sq[:, mask].sum(axis=1)

    non_shared = np.where(assignment != SHARED)[0]
    cur_cov = np.sort(exp_sq, axis=0)[-top_k:, :].sum()
    n_swaps = 0

    rng = np.random.default_rng(rng_seed)
    order = rng.permutation(non_shared)

    # Build expert channel lists for fast lookup
    expert_chs = [list(non_shared[assignment[non_shared] == e]) for e in range(n_experts)]

    for i in order:
        src_i = int(assignment[i])
        sq_i = act_sq[:, i]

        best_cov_swap = cur_cov
        best_j = -1
        best_src_j = -1

        for dst_e in range(n_experts):
            if dst_e == src_i: continue
            chs_j = expert_chs[dst_e]
            if len(chs_j) == 0: continue
            # Sample candidates
            if len(chs_j) <= n_candidates:
                candidates = chs_j
            else:
                idx = rng.integers(0, len(chs_j), size=n_candidates)
                candidates = [chs_j[k] for k in idx]

            for j in candidates:
                sq_j = act_sq[:, j]
                trial = exp_sq.copy()
                trial[src_i] = exp_sq[src_i] - sq_i + sq_j
                trial[dst_e] = exp_sq[dst_e] - sq_j + sq_i
                cov = np.sort(trial, axis=0)[-top_k:, :].sum()
                if cov > best_cov_swap:
                    best_cov_swap = cov
                    best_j = j
                    best_src_j = dst_e

        if best_j >= 0:
            sq_j = act_sq[:, best_j]
            exp_sq[src_i] = exp_sq[src_i] - sq_i + sq_j
            exp_sq[best_src_j] = exp_sq[best_src_j] - sq_j + sq_i
            # Update expert channel lists
            expert_chs[src_i].remove(int(i))
            expert_chs[src_i].append(int(best_j))
            expert_chs[best_src_j].remove(int(best_j))
            expert_chs[best_src_j].append(int(i))
            assignment[i] = best_src_j
            assignment[best_j] = src_i
            cur_cov = best_cov_swap
            n_swaps += 1

    return n_swaps, assignment, cur_cov


out = open('examples/grouping-search/swapsweep3_results.txt', 'w')
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

    for n_cand in [1, 2, 3]:
        rng = np.random.default_rng(0)
        t0 = time.time()
        best_ps = -1
        best_assign = None
        n_completed = 0

        while time.time() - t0 < 22:
            sub = np.random.default_rng(rng.integers(0, 2**31) + n_completed)
            assign = make_seed(act, imp, n_experts, shared_ratio, sub)

            proxy = 0
            prev_proxy = -1

            # Standard sweeps until convergence
            for sw in range(60):
                n_moves, assign, proxy = sweep_standard(act, assign, n_experts, 2, n_completed*100+sw, max_size)
                if n_moves == 0: break

            # Swap + standard cycles
            for phase in range(10):
                n_swaps, assign, proxy_new = sweep_swap_fast(
                    act, assign, n_experts, 2, n_completed*1000+phase, max_size, n_candidates=n_cand)
                if proxy_new > proxy:
                    proxy = proxy_new

                if n_swaps > 0:
                    for sw in range(30):
                        n_moves, assign, proxy2 = sweep_standard(
                            act, assign, n_experts, 2, n_completed*10000+phase*30+sw, max_size)
                        if proxy2 > proxy: proxy = proxy2
                        if n_moves == 0: break
                else:
                    break

            if proxy > best_ps:
                best_ps = proxy
                best_assign = assign.copy()
            n_completed += 1

        err = oracle_topk_error(activations=act, down=down, assignment=best_assign, top_k=2)
        counts = np.bincount(best_assign[best_assign != SHARED], minlength=n_experts)
        log(f'  n_cand={n_cand}: err={err:.6f}, proxy={best_ps:.2f}, completed={n_completed}, t={time.time()-t0:.1f}s')

log('\nDone.')
out.close()
