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

norms_sq = (contribs ** 2).sum(axis=2)
down_sq = (down ** 2).sum(axis=0)
act_sq = act ** 2
dense = act @ down.T
dense_norms = np.linalg.norm(dense, axis=1)
shared_mask_g = assign == SHARED
shared_contrib = act[:, shared_mask_g] @ down[:, shared_mask_g].T
norms_T = np.sqrt(np.maximum(norms_sq, 0)).T
selected = np.argsort(-norms_T, axis=1)[:, :top_k]
selected_mask = np.zeros((T, n_experts), dtype=bool)
selected_mask[np.arange(T), selected[:, 0]] = True
selected_mask[np.arange(T), selected[:, 1]] = True

non_shared = np.where(assign != SHARED)[0]
i = non_shared[0]
src = int(assign[i])
a_i = act[:, i]
d_i = down[:, i]
d_sq_i = down_sq[i]
a_sq_i = act_sq[:, i]

p_all = contribs @ d_i
p_src = p_all[src]

new_sq_src = norms_sq[src] - 2*a_i*p_src + a_sq_i*d_sq_i
new_norm_src = np.sqrt(np.maximum(new_sq_src, 0))

dst = (src+1) % n_experts
p_dst = p_all[dst]
new_sq_dst = norms_sq[dst] + 2*a_i*p_dst + a_sq_i*d_sq_i
new_norm_dst = np.sqrt(np.maximum(new_sq_dst, 0))

new_norms_T = norms_T.copy()
new_norms_T[:, src] = new_norm_src
new_norms_T[:, dst] = new_norm_dst
new_selected = np.argsort(-new_norms_T, axis=1)[:, :top_k]
new_selected_mask = np.zeros((T, n_experts), dtype=bool)
new_selected_mask[np.arange(T), new_selected[:, 0]] = True
new_selected_mask[np.arange(T), new_selected[:, 1]] = True

# All affected tokens: where src or dst is in old or new selection
affected = (selected_mask[:, src] | selected_mask[:, dst] |
            new_selected_mask[:, src] | new_selected_mask[:, dst])
aff_idx = np.where(affected)[0]
print(f'Affected tokens: {len(aff_idx)} (changed: {(new_selected_mask != selected_mask).any(axis=1).sum()})')

# Compute new recon for affected tokens
a_aff = a_i[aff_idx]
delta_aff = a_aff[:, np.newaxis] * d_i[np.newaxis, :]  # [n_aff, H]

new_r = shared_contrib[aff_idx].copy()
for e in range(n_experts):
    e_mask = new_selected_mask[aff_idx, e]
    if not e_mask.any():
        continue
    if e == src:
        new_r[e_mask] += contribs[src][aff_idx[e_mask]] - delta_aff[e_mask]
    elif e == dst:
        new_r[e_mask] += contribs[dst][aff_idx[e_mask]] + delta_aff[e_mask]
    else:
        new_r[e_mask] += contribs[e][aff_idx[e_mask]]

# Build initial recon
recon = shared_contrib.copy()
for e in range(n_experts):
    tok_mask = selected_mask[:, e]
    if tok_mask.any():
        recon[tok_mask] += contribs[e, tok_mask]

new_errors_aff = np.linalg.norm(dense[aff_idx] - new_r, axis=1) / (dense_norms[aff_idx] + 1e-12)
errors_init = np.linalg.norm(dense - recon, axis=1) / (dense_norms + 1e-12)
new_errors = errors_init.copy()
new_errors[aff_idx] = new_errors_aff
new_mean_err = new_errors.mean()

# Compare with oracle
assign_test = assign.copy()
assign_test[i] = dst
err_oracle = oracle_topk_error(activations=act, down=down, assignment=assign_test, top_k=2)

print(f'Maintained: {new_mean_err:.6f}, Oracle: {err_oracle:.6f}')
print(f'Match: {abs(new_mean_err - err_oracle) < 1e-6}')
print(f'Diff: {abs(new_mean_err - err_oracle):.2e}')
