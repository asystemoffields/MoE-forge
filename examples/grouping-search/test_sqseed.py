"""Test sq-acts k-means with intelligent balance repair."""
import sys, time
sys.path.insert(0, 'src')
import numpy as np
from moeforge.grouping import oracle_topk_error, SHARED

import importlib.util
spec = importlib.util.spec_from_file_location('c', 'examples/grouping-search/candidates/seed_clustering.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

d3 = np.load('examples/grouping-search/layer3.npz')
d9 = np.load('examples/grouping-search/layer9.npz')
layers = [('layer3', d3), ('layer9', d9)]

n_experts = 8


def act_sweep_balanced(act, assign, n_experts, top_k, rng_seed, max_size):
    T, I = act.shape
    act_sq = act ** 2
    expert_sq = np.zeros((n_experts, T))
    expert_size = np.zeros(n_experts, dtype=int)
    for e in range(n_experts):
        mask = assign == e
        expert_size[e] = mask.sum()
        if mask.any():
            expert_sq[e] = act_sq[:, mask].sum(axis=1)
    non_shared = np.where(assign != SHARED)[0]
    order = np.random.default_rng(rng_seed).permutation(non_shared)
    n_moves = 0
    cur_topk = np.sort(expert_sq, axis=0)[-2:, :].sum()
    for i in order:
        src = int(assign[i])
        sq_i = act_sq[:, i]
        new_sq_src = expert_sq[src] - sq_i
        best_cov = cur_topk
        best_dst = src
        for dst in range(n_experts):
            if dst == src:
                continue
            if expert_size[dst] >= max_size:
                continue
            if expert_size[src] <= 1:
                continue
            trial = expert_sq.copy()
            trial[src] = new_sq_src
            trial[dst] = expert_sq[dst] + sq_i
            topk = np.sort(trial, axis=0)[-2:, :].sum()
            if topk > best_cov:
                best_cov = topk
                best_dst = dst
        if best_dst != src:
            expert_sq[src] = new_sq_src
            expert_sq[best_dst] = expert_sq[best_dst] + sq_i
            expert_size[src] -= 1
            expert_size[best_dst] += 1
            cur_topk = best_cov
            assign[i] = best_dst
            n_moves += 1
    return n_moves, assign


def proxy_score(act, assign, n_experts, top_k=2):
    T = act.shape[0]
    exp_sq = np.zeros((n_experts, T))
    for e in range(n_experts):
        mask = assign == e
        if mask.any():
            exp_sq[e] = (act[:, mask]**2).sum(axis=1)
    return np.sort(exp_sq, axis=0)[-top_k:, :].sum()


def make_sq_seed_balanced(act, imp, n_experts, shared_ratio, rng, max_size):
    I = imp.shape[0]
    n_shared = int(round(shared_ratio * I))
    order = np.argsort(-imp)
    assign = np.empty(I, dtype=int)
    assign[order[:n_shared]] = SHARED
    remaining = order[n_shared:]

    # Squared activation k-means
    sq_vecs = (act[:, remaining]**2).T
    sq_vecs = sq_vecs / (np.linalg.norm(sq_vecs, axis=1, keepdims=True) + 1e-12)
    labels = mod._kmeans(sq_vecs, n_experts, rng)
    for ch, lab in zip(remaining, labels):
        assign[ch] = int(lab)

    # Repair balance: move lowest-energy channels from oversized to undersized experts
    channel_energy = (act[:, remaining]**2).mean(axis=0)  # [n_rem]
    # mapping: remaining[j] -> energy j
    ch_to_en = dict(zip(remaining, channel_energy))

    for _ in range(10000):
        counts = np.bincount(assign[assign != SHARED], minlength=n_experts)
        if counts.max() <= max_size:
            break
        worst_src = int(np.argmax(counts))
        src_chs = remaining[assign[remaining] == worst_src]
        energies = np.array([ch_to_en[c] for c in src_chs])
        lowest_idx = int(np.argmin(energies))
        ch = src_chs[lowest_idx]
        dst = int(np.argmin(counts))
        assign[ch] = dst

    return assign


t0 = time.time()
for layer_name, data in layers:
    act = data['activations'].astype(np.float64)
    imp = data['importance'].astype(np.float64)
    down = data['down'].astype(np.float64)
    I = imp.shape[0]
    n_routed = I - int(round(0.125 * I))
    max_size = int(2.0 * n_routed / n_experts)

    t_layer = time.time()
    best_ps = -1
    best_assign = None

    for try_idx in range(10):
        if time.time() - t_layer > 20:
            break
        rng = np.random.default_rng(try_idx * 17)
        assign = make_sq_seed_balanced(act, imp, n_experts, 0.125, rng, max_size)

        for sweep in range(60):
            n_moves, assign = act_sweep_balanced(
                act, assign, n_experts, 2, try_idx*100+sweep, max_size)
            if n_moves == 0:
                break

        ps = proxy_score(act, assign, n_experts)
        if ps > best_ps:
            best_ps = ps
            best_assign = assign.copy()

    err = oracle_topk_error(activations=act, down=down, assignment=best_assign, top_k=2)
    counts = np.bincount(best_assign[best_assign != SHARED], minlength=n_experts)
    print(f'{layer_name}: err={err:.6f}, sizes={counts.tolist()}, '
          f'max/ideal={counts.max()/(n_routed/n_experts):.2f}, '
          f'time={time.time()-t_layer:.1f}s')

print(f'Total: {time.time()-t0:.1f}s')
