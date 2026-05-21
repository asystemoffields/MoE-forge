"""Test adaptive n_candidates approach."""
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


def sweep_swap(activations, assignment, n_experts, top_k, rng_seed, n_candidates=8):
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
            if not chs_j: continue
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
            expert_chs[src_i].remove(int(i))
            expert_chs[src_i].append(int(best_j))
            expert_chs[best_src_j].remove(int(best_j))
            expert_chs[best_src_j].append(int(i))
            assignment[i] = best_src_j
            assignment[best_j] = src_i
            cur_cov = best_cov_swap
            n_swaps += 1
    return n_swaps, assignment, cur_cov


def run_adaptive(act, imp, n_experts, shared_ratio, rng, time_limit=30.0):
    """Adaptive: use different n_candidates values in swap phases."""
    T, I = act.shape
    n_shared = int(round(shared_ratio * I))
    n_routed = I - n_shared
    max_size = int(2.0 * n_routed / n_experts)

    best_assign = None
    best_proxy = -1.0
    t_start = time.time()
    trial = 0

    while time.time() - t_start < time_limit:
        sub = np.random.default_rng(rng.integers(0, 2**31) + trial)
        assign = make_seed(act, imp, n_experts, shared_ratio, sub)

        proxy = 0.0
        for sw in range(60):
            n_moves, assign, proxy = sweep_standard(act, assign, n_experts, 2, trial*100+sw, max_size)
            if n_moves == 0: break

        # Try multiple n_candidates values: 6, 10 (widening search)
        for cand_idx, n_cand in enumerate([6, 10]):
            for sp in range(5):
                n_sw, assign, pn = sweep_swap(act, assign, n_experts, 2, trial*1000+cand_idx*100+sp, n_cand)
                if pn > proxy: proxy = pn
                if n_sw == 0: break
                for sw in range(30):
                    n_moves, assign, p2 = sweep_standard(act, assign, n_experts, 2, trial*100000+cand_idx*1000+sp*30+sw, max_size)
                    if p2 > proxy: proxy = p2
                    if n_moves == 0: break

        if proxy > best_proxy:
            best_proxy = proxy
            best_assign = assign.copy()
        trial += 1

    return best_assign, best_proxy, trial


out = open('examples/grouping-search/adaptive_results.txt', 'w')
def log(s):
    out.write(s + '\n')
    out.flush()

d3 = np.load('examples/grouping-search/layer3.npz')
d9 = np.load('examples/grouping-search/layer9.npz')
layers = [('layer3', d3), ('layer9', d9)]

for layer_name, data in layers:
    act = data['activations'].astype(np.float64)
    imp = data['importance'].astype(np.float64)
    down = data['down'].astype(np.float64)
    log(f'\n=== {layer_name} ===')
    for tl in [22, 30, 45]:
        rng = np.random.default_rng(0)
        t0 = time.time()
        best, proxy, n_trials = run_adaptive(act, imp, 8, 0.125, rng, tl)
        dt = time.time() - t0
        err = oracle_topk_error(activations=act, down=down, assignment=best, top_k=2)
        log(f'  adaptive(6,10), tl={tl}s: err={err:.6f}, proxy={proxy:.2f}, trials={n_trials}, dt={dt:.1f}s')

log('\nDone.')
out.close()
