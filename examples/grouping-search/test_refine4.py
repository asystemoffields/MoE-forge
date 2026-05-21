"""Test hybrid proxy + exact oracle verification approach."""
import sys, time
sys.path.insert(0, 'src')
import numpy as np
from moeforge.grouping import oracle_topk_error, SHARED

d3 = np.load('examples/grouping-search/layer3.npz')
act = d3['activations'].astype(np.float64)
imp = d3['importance'].astype(np.float64)
down = d3['down'].astype(np.float64)
T, I = act.shape
H = down.shape[0]
n_experts = 8
top_k = 2

import importlib.util
spec = importlib.util.spec_from_file_location('c', 'examples/grouping-search/candidates/seed_clustering.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
rng = np.random.default_rng(0)
assign0 = mod.group({'importance': imp, 'activations': act}, 8, 0.125, rng)


def build_state(act, down, assign, n_experts, top_k):
    T, I = act.shape
    H = down.shape[0]
    contribs = np.zeros((n_experts, T, H))
    for e in range(n_experts):
        mask = assign == e
        if mask.any():
            contribs[e] = act[:, mask] @ down[:, mask].T
    norms_sq = (contribs ** 2).sum(axis=2)
    shared_mask_g = assign == SHARED
    shared_contrib = act[:, shared_mask_g] @ down[:, shared_mask_g].T if shared_mask_g.any() else np.zeros((T, H))
    norms_T = np.sqrt(np.maximum(norms_sq, 0)).T
    selected = np.argsort(-norms_T, axis=1)[:, :top_k]
    sel_mask = np.zeros((T, n_experts), dtype=bool)
    sel_mask[np.arange(T), selected[:, 0]] = True
    sel_mask[np.arange(T), selected[:, 1]] = True
    recon = shared_contrib.copy()
    for e in range(n_experts):
        tok_mask = sel_mask[:, e]
        if tok_mask.any():
            recon[tok_mask] += contribs[e, tok_mask]
    dense = act @ down.T
    dense_norms = np.linalg.norm(dense, axis=1)
    errors = np.linalg.norm(dense - recon, axis=1) / (dense_norms + 1e-12)
    return contribs, norms_sq, shared_contrib, norms_T, selected, sel_mask, recon, dense, dense_norms, errors


def run_hybrid_sweep(act, down, assign, contribs, norms_sq, shared_contrib,
                     norms_T, selected, sel_mask, dense, dense_norms, errors,
                     n_experts, top_k, rng_seed):
    T, I = act.shape
    H = down.shape[0]
    act_sq = act ** 2
    down_sq = (down ** 2).sum(axis=0)
    curr_err = float(errors.mean())
    non_shared = np.where(assign != SHARED)[0]
    channel_order = np.random.default_rng(rng_seed).permutation(non_shared)
    n_moves = 0

    for i in channel_order:
        src = int(assign[i])
        a_i = act[:, i]
        d_i = down[:, i]
        d_sq_i = down_sq[i]
        a_sq_i = act_sq[:, i]

        p_all = contribs @ d_i  # [E, T] -- key computation
        p_src = p_all[src]
        new_sq_src = norms_sq[src] - 2*a_i*p_src + a_sq_i*d_sq_i
        new_norm_src = np.sqrt(np.maximum(new_sq_src, 0))

        # Proxy: top-k norm sum over all trial dsts
        new_sq_all_dst = norms_sq + 2*a_i[np.newaxis,:]*p_all + (a_sq_i*d_sq_i)[np.newaxis,:]
        new_sq_base = norms_sq.copy()
        new_sq_base[src] = new_sq_src

        cur_topk_sum = np.sort(norms_T, axis=1)[:, -top_k:].sum()
        best_proxy = cur_topk_sum
        best_dst = src

        for dst in range(n_experts):
            if dst == src:
                continue
            trial_sq = new_sq_base.copy()
            trial_sq[dst] = new_sq_all_dst[dst]
            trial_topk = np.sort(np.sqrt(np.maximum(trial_sq, 0)).T, axis=1)[:, -top_k:].sum()
            if trial_topk > best_proxy:
                best_proxy = trial_topk
                best_dst = dst

        if best_dst == src:
            continue  # Proxy says no move is beneficial

        # Verify best_dst with exact oracle error
        dst = best_dst
        p_dst = p_all[dst]
        new_sq_dst = norms_sq[dst] + 2*a_i*p_dst + a_sq_i*d_sq_i
        new_norm_dst = np.sqrt(np.maximum(new_sq_dst, 0))
        new_norms_T = norms_T.copy()
        new_norms_T[:, src] = new_norm_src
        new_norms_T[:, dst] = new_norm_dst
        new_sel = np.argsort(-new_norms_T, axis=1)[:, :top_k]
        new_sel_mask = np.zeros((T, n_experts), dtype=bool)
        new_sel_mask[np.arange(T), new_sel[:, 0]] = True
        new_sel_mask[np.arange(T), new_sel[:, 1]] = True

        affected = (sel_mask[:, src] | sel_mask[:, dst] |
                    new_sel_mask[:, src] | new_sel_mask[:, dst])
        aff_idx = np.where(affected)[0]

        if len(aff_idx) == 0:
            continue

        a_aff = a_i[aff_idx]
        delta_aff = a_aff[:, np.newaxis] * d_i[np.newaxis, :]
        new_r = shared_contrib[aff_idx].copy()
        for e in range(n_experts):
            e_mask = new_sel_mask[aff_idx, e]
            if not e_mask.any():
                continue
            if e == src:
                new_r[e_mask] += contribs[src][aff_idx[e_mask]] - delta_aff[e_mask]
            elif e == dst:
                new_r[e_mask] += contribs[dst][aff_idx[e_mask]] + delta_aff[e_mask]
            else:
                new_r[e_mask] += contribs[e][aff_idx[e_mask]]

        new_errs_aff = np.linalg.norm(dense[aff_idx] - new_r, axis=1) / (dense_norms[aff_idx] + 1e-12)
        new_err_i = float(errors.sum() - errors[aff_idx].sum() + new_errs_aff.sum()) / T

        if new_err_i < curr_err:
            delta = a_i[:, np.newaxis] * d_i[np.newaxis, :]
            contribs[src] -= delta
            contribs[dst] += delta
            norms_sq[src] = new_sq_src
            norms_sq[dst] = new_sq_dst
            norms_T = new_norms_T
            errors[aff_idx] = new_errs_aff
            curr_err = new_err_i
            selected = new_sel
            sel_mask = new_sel_mask
            assign[i] = dst
            n_moves += 1

    return n_moves, curr_err, contribs, norms_sq, norms_T, selected, sel_mask, errors


# Run full refinement
assign = assign0.copy()
state = build_state(act, down, assign, n_experts, top_k)
contribs, norms_sq, shared_contrib, norms_T, selected, sel_mask, recon, dense, dense_norms, errors = state

t_total = time.time()
for sweep in range(20):
    t0 = time.time()
    n_moves, curr_err, contribs, norms_sq, norms_T, selected, sel_mask, errors = run_hybrid_sweep(
        act, down, assign, contribs, norms_sq, shared_contrib,
        norms_T, selected, sel_mask, dense, dense_norms, errors,
        n_experts, top_k, rng_seed=sweep
    )
    dt = time.time() - t0
    err_true = oracle_topk_error(activations=act, down=down, assignment=assign, top_k=2)
    print(f'Sweep {sweep+1}: {n_moves} moves, maintained={curr_err:.6f}, oracle={err_true:.6f}, {dt:.2f}s')

    if n_moves == 0:
        break
    if time.time() - t_total > 45:
        print('Time limit')
        break

print(f'Total: {time.time()-t_total:.1f}s')
