"""Test smart balance repair from unconstrained sq-acts k-means.

Key idea: move channels from oversized experts to undersized ones,
prioritizing channels with the lowest 'specificity' (those whose activation
is most uniform across tokens, so moving them minimally changes which
expert is top-2 for each token).
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


def smart_balance_repair(activations, assignment, n_experts, max_size, remaining, specificity_mode='cv'):
    """Move channels from oversized to undersized experts.

    specificity_mode:
    - 'energy': move lowest total energy channel (current approach)
    - 'cv': move channel with lowest coefficient of variation (most uniform)
    - 'min_proxy_loss': try each candidate and pick the one with least proxy loss
    """
    act_sq = activations ** 2

    # Build expert sq sums
    T = activations.shape[0]
    exp_sq = np.zeros((n_experts, T))
    for e in range(n_experts):
        mask = assignment == e
        if mask.any():
            exp_sq[e] = act_sq[:, mask].sum(axis=1)

    # Precompute channel properties
    ch_act_sq = act_sq[:, remaining]  # [T, n_rem]

    if specificity_mode == 'cv':
        # Coefficient of variation: std / mean of sq-activation
        ch_mean = ch_act_sq.mean(axis=0)  # [n_rem]
        ch_std = ch_act_sq.std(axis=0)    # [n_rem]
        ch_score = ch_std / (ch_mean + 1e-12)  # high = specific, low = generic
    elif specificity_mode == 'energy':
        ch_score = ch_act_sq.sum(axis=0)   # high = high energy
    elif specificity_mode == 'min_proxy_loss':
        ch_score = ch_act_sq.sum(axis=0)   # fallback initial

    ch_score_map = dict(zip(remaining.tolist(), ch_score.tolist()))

    cur_proxy = np.sort(exp_sq, axis=0)[-2:, :].sum()

    for _ in range(100000):
        counts = np.bincount(assignment[remaining], minlength=n_experts)
        if counts.max() <= max_size:
            break
        src = int(np.argmax(counts))
        dst = int(np.argmin(counts))
        src_chs = remaining[assignment[remaining] == src]

        if specificity_mode == 'min_proxy_loss':
            # Try moving each channel, pick the one with minimum proxy loss
            best_loss = np.inf
            best_ch = None
            for ch in src_chs:
                sq_i = act_sq[:, ch]
                trial = exp_sq.copy()
                trial[src] = exp_sq[src] - sq_i
                trial[dst] = exp_sq[dst] + sq_i
                new_proxy = np.sort(trial, axis=0)[-2:, :].sum()
                loss = cur_proxy - new_proxy
                if loss < best_loss:
                    best_loss = loss
                    best_ch = ch
            ch = best_ch
            sq_i = act_sq[:, ch]
            exp_sq[src] -= sq_i
            exp_sq[dst] += sq_i
            cur_proxy = np.sort(exp_sq, axis=0)[-2:, :].sum()
        else:
            scores = np.array([ch_score_map[c] for c in src_chs])
            lowest_idx = int(np.argmin(scores))
            ch = src_chs[lowest_idx]
            sq_i = act_sq[:, ch]
            exp_sq[src] -= sq_i
            exp_sq[dst] += sq_i

        assignment[ch] = dst

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
    n_shared = int(round(shared_ratio * I))
    n_routed = I - n_shared
    max_size = int(2.0 * n_routed / n_experts)
    print(f'\n=== {layer_name} ===')

    order = np.argsort(-imp)
    remaining = order[n_shared:]

    for mode in ['energy', 'cv', 'min_proxy_loss']:
        # Unconstrained sq-acts k-means (best seed)
        rng = np.random.default_rng(4)  # seed 4 was best for layer3 (err=0.362)
        sq_vecs = (act[:, remaining] ** 2).T
        sq_vecs = sq_vecs / (np.linalg.norm(sq_vecs, axis=1, keepdims=True) + 1e-12)
        labels = _kmeans(sq_vecs, n_experts, rng)
        assign = np.empty(I, dtype=int)
        assign[order[:n_shared]] = SHARED
        for ch, lab in zip(remaining, labels):
            assign[ch] = int(lab)

        t0 = time.time()
        assign = smart_balance_repair(act, assign, n_experts, max_size, remaining, specificity_mode=mode)
        counts = np.bincount(assign[assign != SHARED], minlength=n_experts)
        err = oracle_topk_error(activations=act, down=down, assignment=assign, top_k=2)
        print(f'  After {mode} repair (no sweep): err={err:.6f}, sizes={counts.tolist()}, valid={counts.max()<=max_size}, t={time.time()-t0:.2f}s')

        # Now run sweep
        for sweep in range(60):
            n_moves, assign, proxy = sweep_fast(act, assign, n_experts, 2, sweep, max_size)
            if n_moves == 0: break
        err2 = oracle_topk_error(activations=act, down=down, assignment=assign, top_k=2)
        counts2 = np.bincount(assign[assign != SHARED], minlength=n_experts)
        print(f'  After {mode} repair + sweep: err={err2:.6f}, sizes={counts2.tolist()}')

print('\nDone.')
