"""Test a fast approach:
Instead of computing the full oracle error per trial move,
use a smarter proxy that INCLUDES the recon change for src/dst-affected tokens.

Key insight: For moving channel i from src to dst,
the oracle error changes mainly for tokens where src or dst is selected.
We can compute this VECTORIZED across all dst candidates.
"""
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

assign = assign0.copy()
contribs = np.zeros((n_experts, T, H))
for e in range(n_experts):
    mask = assign == e
    if mask.any():
        contribs[e] = act[:, mask] @ down[:, mask].T

norms_sq = (contribs ** 2).sum(axis=2)  # [E, T]
down_sq = (down ** 2).sum(axis=0)
act_sq = act ** 2
dense = act @ down.T  # [T, H]
dense_norms = np.linalg.norm(dense, axis=1)
shared_mask_g = assign == SHARED
shared_contrib = act[:, shared_mask_g] @ down[:, shared_mask_g].T

norms_T = np.sqrt(np.maximum(norms_sq, 0)).T  # [T, E]
selected = np.argsort(-norms_T, axis=1)[:, :top_k]  # [T, top_k]
selected_mask = np.zeros((T, n_experts), dtype=bool)
selected_mask[np.arange(T), selected[:, 0]] = True
selected_mask[np.arange(T), selected[:, 1]] = True

recon = shared_contrib.copy()
for e in range(n_experts):
    tok_mask = selected_mask[:, e]
    if tok_mask.any():
        recon[tok_mask] += contribs[e, tok_mask]

residual = dense - recon  # [T, H]
errors = np.linalg.norm(residual, axis=1) / (dense_norms + 1e-12)
curr_err = float(errors.mean())
print(f'Initial error: {curr_err:.6f}')

non_shared = np.where(assign != SHARED)[0]

# Time accurate per-channel evaluation for one channel
i = non_shared[0]
src = int(assign[i])
a_i = act[:, i]
d_i = down[:, i]
d_sq_i = down_sq[i]
a_sq_i = act_sq[:, i]

t0 = time.time()
for _ in range(100):
    p_all = contribs @ d_i  # [E, T]: key computation
    p_src = p_all[src]
    new_sq_src = norms_sq[src] - 2*a_i*p_src + a_sq_i*d_sq_i
    new_norm_src = np.sqrt(np.maximum(new_sq_src, 0))

    best_err_i = curr_err
    best_dst_i = src

    for dst in range(n_experts):
        if dst == src:
            continue
        p_dst = p_all[dst]
        new_sq_dst = norms_sq[dst] + 2*a_i*p_dst + a_sq_i*d_sq_i
        new_norm_dst = np.sqrt(np.maximum(new_sq_dst, 0))

        new_norms_T = norms_T.copy()
        new_norms_T[:, src] = new_norm_src
        new_norms_T[:, dst] = new_norm_dst
        new_selected = np.argsort(-new_norms_T, axis=1)[:, :top_k]
        new_sel_mask = np.zeros((T, n_experts), dtype=bool)
        new_sel_mask[np.arange(T), new_selected[:, 0]] = True
        new_sel_mask[np.arange(T), new_selected[:, 1]] = True

        affected = (selected_mask[:, src] | selected_mask[:, dst] |
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

        if new_err_i < best_err_i:
            best_err_i = new_err_i
            best_dst_i = dst

dt = time.time() - t0
print(f'100 channel evals: {dt*1000:.0f}ms = {dt/100*1000:.2f}ms each')
print(f'Estimated for 1344 channels: {dt/100*1344:.2f}s per sweep')
