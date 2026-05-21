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


def run_refinement_exact(act, down, assign_init, n_experts=8, top_k=2, n_sweeps=30, time_limit=25, verbose=True):
    T, I = act.shape
    H = down.shape[0]
    assign = assign_init.copy()

    contribs = np.zeros((n_experts, T, H))
    for e in range(n_experts):
        mask = assign == e
        if mask.any():
            contribs[e] = act[:, mask] @ down[:, mask].T

    norms_sq = (contribs ** 2).sum(axis=2)  # [E, T]
    act_sq = act ** 2
    down_sq = (down ** 2).sum(axis=0)

    dense = act @ down.T  # [T, H]
    dense_norms = np.linalg.norm(dense, axis=1)  # [T]
    shared_mask = assign == SHARED
    shared_contrib = act[:, shared_mask] @ down[:, shared_mask].T if shared_mask.any() else np.zeros((T, H))

    # Initial top-k selection
    norms_T = np.sqrt(np.maximum(norms_sq, 0)).T  # [T, E]
    selected = np.argsort(-norms_T, axis=1)[:, :top_k]  # [T, top_k]

    # selected_mask[t, e] = True if expert e selected for token t
    selected_mask = np.zeros((T, n_experts), dtype=bool)
    for j in range(top_k):
        selected_mask[np.arange(T), selected[:, j]] = True

    # Current reconstruction
    recon = shared_contrib.copy()
    for e in range(n_experts):
        tok_mask = selected_mask[:, e]
        if tok_mask.any():
            recon[tok_mask] += contribs[e, tok_mask]

    errors = np.linalg.norm(dense - recon, axis=1) / (dense_norms + 1e-12)
    curr_err = float(errors.mean())

    non_shared = np.where(assign != SHARED)[0]
    t_start = time.time()

    for sweep in range(n_sweeps):
        n_moves = 0
        channel_order = np.random.default_rng(sweep).permutation(non_shared)

        for i in channel_order:
            src = int(assign[i])
            a_i = act[:, i]   # [T]
            d_i = down[:, i]  # [H]
            d_sq_i = down_sq[i]
            a_sq_i = act_sq[:, i]

            # On-demand projection for all experts
            p_all = contribs @ d_i  # [E, T]
            p_src = p_all[src]

            new_sq_src = norms_sq[src] - 2*a_i*p_src + a_sq_i*d_sq_i
            new_norm_src = np.sqrt(np.maximum(new_sq_src, 0))

            best_dst = src
            best_err = curr_err
            best_state = None

            for dst in range(n_experts):
                if dst == src:
                    continue

                p_dst = p_all[dst]
                new_sq_dst = norms_sq[dst] + 2*a_i*p_dst + a_sq_i*d_sq_i
                new_norm_dst = np.sqrt(np.maximum(new_sq_dst, 0))

                # New norms_T
                new_norms_T = norms_T.copy()
                new_norms_T[:, src] = new_norm_src
                new_norms_T[:, dst] = new_norm_dst

                # New top-k selection
                new_selected = np.argsort(-new_norms_T, axis=1)[:, :top_k]
                new_selected_mask = np.zeros((T, n_experts), dtype=bool)
                new_selected_mask[np.arange(T), new_selected[:, 0]] = True
                new_selected_mask[np.arange(T), new_selected[:, 1]] = True

                # Changed tokens
                changed = (new_selected_mask != selected_mask).any(axis=1)
                chg_idx = np.where(changed)[0]

                if len(chg_idx) == 0:
                    continue

                # Compute new errors for changed tokens
                new_errors = errors.copy()

                # Batch compute new recon for changed tokens
                # Channel i delta contribution: a_i[t] * d_i
                a_chg = a_i[chg_idx]  # [n_chg]
                delta_chg = a_chg[:, np.newaxis] * d_i[np.newaxis, :]  # [n_chg, H]

                new_r = shared_contrib[chg_idx].copy()  # [n_chg, H]
                for e in range(n_experts):
                    e_mask = new_selected_mask[chg_idx, e]  # [n_chg]
                    if not e_mask.any():
                        continue
                    if e == src:
                        new_r[e_mask] += contribs[src][chg_idx[e_mask]] - delta_chg[e_mask]
                    elif e == dst:
                        new_r[e_mask] += contribs[dst][chg_idx[e_mask]] + delta_chg[e_mask]
                    else:
                        new_r[e_mask] += contribs[e][chg_idx[e_mask]]

                new_errs_chg = np.linalg.norm(dense[chg_idx] - new_r, axis=1) / (dense_norms[chg_idx] + 1e-12)
                new_errors[chg_idx] = new_errs_chg
                new_mean_err = float(new_errors.mean())

                if new_mean_err < best_err:
                    best_err = new_mean_err
                    best_dst = dst
                    best_state = (new_sq_src, new_sq_dst, new_norm_src, new_norm_dst,
                                  new_norms_T, new_selected, new_selected_mask, new_errors)

            if best_dst != src and best_state is not None:
                bsq_src, bsq_dst, bnorm_src, bnorm_dst, bnorms_T, bsel, bsel_mask, berrs = best_state

                # Apply move
                delta = a_i[:, np.newaxis] * d_i[np.newaxis, :]
                contribs[src] -= delta
                contribs[best_dst] += delta
                norms_sq[src] = bsq_src
                norms_sq[best_dst] = bsq_dst
                norms_T = bnorms_T

                # Update recon for changed tokens
                chg = (bsel_mask != selected_mask).any(axis=1)
                chg_idx = np.where(chg)[0]
                for t in chg_idx:
                    r = shared_contrib[t].copy()
                    for e in range(n_experts):
                        if bsel_mask[t, e]:
                            r += contribs[e, t]
                    recon[t] = r

                selected = bsel
                selected_mask = bsel_mask
                errors = berrs
                curr_err = best_err
                assign[i] = best_dst
                n_moves += 1

        if verbose:
            print(f'  Sweep {sweep+1}: {n_moves} moves, err={curr_err:.6f}, elapsed={time.time()-t_start:.1f}s')

        if n_moves == 0:
            break
        if time.time() - t_start > time_limit:
            break

    return assign


rng = np.random.default_rng(0)
assign_init = mod.group({'importance': imp, 'activations': act}, 8, 0.125, rng)
t0 = time.time()
assign_refined = run_refinement_exact(act, down, assign_init, time_limit=25)
err = oracle_topk_error(activations=act, down=down, assignment=assign_refined, top_k=2)
print(f'Final oracle error: {err:.6f}, total: {time.time()-t0:.1f}s')
