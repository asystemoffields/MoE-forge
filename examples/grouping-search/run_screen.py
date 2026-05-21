"""Multi-fidelity: screen many seeds with fast sweep, optimize top seeds with swap."""
import sys, time
sys.path.insert(0, 'src')
import numpy as np
from moeforge.grouping import oracle_topk_error

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


def _make_seed(activations, importance, n_experts, shared_ratio, rng):
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


def _sweep(activations, assignment, n_experts, top_k, rng_seed, max_size):
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
            if dst == src:
                continue
            if exp_sz[dst] >= max_size:
                continue
            if exp_sz[src] <= 1:
                continue
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


def _swap_sweep(activations, assignment, n_experts, top_k, rng_seed, n_candidates):
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
            if dst_e == src_i:
                continue
            chs_j = expert_chs[dst_e]
            if not chs_j:
                continue
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


def screen_seed(act, assign, n_experts, max_size, trial):
    """Quick screen: just standard sweep until convergence."""
    proxy = 0.0
    for sw in range(60):
        n_moves, assign, proxy = _sweep(act, assign, n_experts, 2, trial * 100 + sw, max_size)
        if n_moves == 0:
            break
    return assign, proxy


def full_opt(act, assign, n_experts, max_size, trial):
    """Full optimization: swap sweeps after standard sweep."""
    proxy = 0.0
    for sw in range(60):
        n_moves, assign, proxy = _sweep(act, assign, n_experts, 2, trial * 100 + sw, max_size)
        if n_moves == 0:
            break
    for sp in range(10):
        n_sw, assign, pn = _swap_sweep(act, assign, n_experts, 2, trial * 1000 + sp, 6)
        if pn > proxy:
            proxy = pn
        if n_sw == 0:
            break
        for sw in range(30):
            n_moves, assign, p2 = _sweep(act, assign, n_experts, 2, trial * 10000 + sp * 30 + sw, max_size)
            if p2 > proxy:
                proxy = p2
            if n_moves == 0:
                break
    return assign, proxy


def run_screened(act, imp, n_experts, shared_ratio, rng, n_screen, top_k_screen, time_limit=30.0):
    """Screen n_screen seeds, fully optimize top top_k_screen."""
    T, I = act.shape
    n_shared = int(round(shared_ratio * I))
    n_routed = I - n_shared
    max_size = int(2.0 * n_routed / n_experts)

    # Phase 1: Screen many seeds quickly
    screened = []
    for trial in range(n_screen):
        sub_rng = np.random.default_rng(rng.integers(0, 2**31) + trial)
        assign = _make_seed(act, imp, n_experts, shared_ratio, sub_rng)
        assign, proxy = screen_seed(act, assign, n_experts, max_size, trial)
        screened.append((proxy, assign.copy(), trial))

    screened.sort(key=lambda x: -x[0])
    top_seeds = screened[:top_k_screen]

    # Phase 2: Full optimization on top seeds
    best_assign = None
    best_proxy = -1.0
    for proxy_screen, assign_init, trial in top_seeds:
        # Re-run full optimization from the screened assignment
        assign, proxy = full_opt(act, assign_init, n_experts, max_size, trial + n_screen * 1000)
        if proxy > best_proxy:
            best_proxy = proxy
            best_assign = assign.copy()

    return best_assign, best_proxy


d3 = np.load('examples/grouping-search/layer3.npz')
d9 = np.load('examples/grouping-search/layer9.npz')

for layer_name, data in [('layer3', d3), ('layer9', d9)]:
    act = data['activations'].astype(np.float64)
    imp = data['importance'].astype(np.float64)
    down = data['down'].astype(np.float64)
    for n_screen, top_k_s in [(10, 3), (15, 3), (20, 3), (10, 5), (20, 5)]:
        rng = np.random.default_rng(0)
        t0 = time.time()
        best, proxy = run_screened(act, imp, 8, 0.125, rng, n_screen, top_k_s, 30.0)
        dt = time.time() - t0
        err = oracle_topk_error(activations=act, down=down, assignment=best, top_k=2)
        print(f'{layer_name} n_screen={n_screen} top={top_k_s}: err={err:.6f}, proxy={proxy:.2f}, dt={dt:.1f}s')
