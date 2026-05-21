"""Oracle repair from multiple sq-acts seeds to find the best starting point."""
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


def oracle_repair_fast(act, down, assignment, n_experts, max_size, remaining, top_n=20):
    T, I = act.shape
    H = down.shape[0]
    dense = act @ down.T
    dense_norms = np.linalg.norm(dense, axis=1)
    shared_mask = assignment == SHARED
    recon_shared = act[:, shared_mask] @ down[:, shared_mask].T if shared_mask.any() else np.zeros((T, H))
    contribs = {}
    for e in range(n_experts):
        mask = assignment == e
        contribs[e] = act[:, mask] @ down[:, mask].T if mask.any() else np.zeros((T, H))
    norms = np.zeros((T, n_experts))
    for e in range(n_experts):
        norms[:, e] = np.linalg.norm(contribs[e], axis=1)
    sel = np.argsort(-norms, axis=1)[:, :2]
    sel_mask = np.zeros((T, n_experts), dtype=bool)
    sel_mask[np.arange(T), sel[:, 0]] = True
    sel_mask[np.arange(T), sel[:, 1]] = True
    recon = recon_shared.copy()
    for e in range(n_experts):
        tok_mask = sel_mask[:, e]
        if tok_mask.any():
            recon[tok_mask] += contribs[e][tok_mask]
    errors = np.linalg.norm(dense - recon, axis=1) / (dense_norms + 1e-12)
    curr_err = float(errors.mean())
    act_sq_all = (act ** 2).sum(axis=0)
    for _ in range(100000):
        counts = np.bincount(assignment[remaining], minlength=n_experts)
        if counts.max() <= max_size:
            break
        src = int(np.argmax(counts))
        dst = int(np.argmin(counts))
        src_chs = remaining[assignment[remaining] == src]
        energies = act_sq_all[src_chs]
        sorted_idx = np.argsort(energies)[:top_n]
        candidates = src_chs[sorted_idx]
        best_delta = np.inf
        best_ch = None
        for ch in candidates:
            a_ch = act[:, ch]
            d_ch = down[:, ch]
            delta_mat = a_ch[:, np.newaxis] * d_ch[np.newaxis, :]
            new_c_src = contribs[src] - delta_mat
            new_c_dst = contribs[dst] + delta_mat
            new_norm_src = np.linalg.norm(new_c_src, axis=1)
            new_norm_dst = np.linalg.norm(new_c_dst, axis=1)
            new_norms = norms.copy()
            new_norms[:, src] = new_norm_src
            new_norms[:, dst] = new_norm_dst
            new_sel = np.argsort(-new_norms, axis=1)[:, :2]
            new_sm = np.zeros((T, n_experts), dtype=bool)
            new_sm[np.arange(T), new_sel[:, 0]] = True
            new_sm[np.arange(T), new_sel[:, 1]] = True
            affected = (sel_mask[:, src] | sel_mask[:, dst] |
                        new_sm[:, src] | new_sm[:, dst])
            aff_idx = np.where(affected)[0]
            if len(aff_idx) == 0:
                new_err = curr_err
            else:
                aff_delta = a_ch[aff_idx, np.newaxis] * d_ch[np.newaxis, :]
                new_r_aff = recon_shared[aff_idx].copy()
                for e2 in range(n_experts):
                    e_mask_aff = new_sm[aff_idx, e2]
                    if not e_mask_aff.any():
                        continue
                    if e2 == src:
                        new_r_aff[e_mask_aff] += contribs[src][aff_idx[e_mask_aff]] - aff_delta[e_mask_aff]
                    elif e2 == dst:
                        new_r_aff[e_mask_aff] += contribs[dst][aff_idx[e_mask_aff]] + aff_delta[e_mask_aff]
                    else:
                        new_r_aff[e_mask_aff] += contribs[e2][aff_idx[e_mask_aff]]
                new_errs_aff = np.linalg.norm(dense[aff_idx] - new_r_aff, axis=1) / (dense_norms[aff_idx] + 1e-12)
                new_err = float(errors.sum() - errors[aff_idx].sum() + new_errs_aff.sum()) / T
            delta_err = new_err - curr_err
            if delta_err < best_delta:
                best_delta = delta_err
                best_ch = ch
        ch = best_ch
        a_ch = act[:, ch]
        d_ch = down[:, ch]
        delta_mat = a_ch[:, np.newaxis] * d_ch[np.newaxis, :]
        contribs[src] -= delta_mat
        contribs[dst] += delta_mat
        norms[:, src] = np.linalg.norm(contribs[src], axis=1)
        norms[:, dst] = np.linalg.norm(contribs[dst], axis=1)
        new_sel = np.argsort(-norms, axis=1)[:, :2]
        sel_mask = np.zeros((T, n_experts), dtype=bool)
        sel_mask[np.arange(T), new_sel[:, 0]] = True
        sel_mask[np.arange(T), new_sel[:, 1]] = True
        sel = new_sel
        recon = recon_shared.copy()
        for e in range(n_experts):
            tok_mask = sel_mask[:, e]
            if tok_mask.any():
                recon[tok_mask] += contribs[e][tok_mask]
        errors = np.linalg.norm(dense - recon, axis=1) / (dense_norms + 1e-12)
        curr_err = float(errors.mean())
        assignment[ch] = dst
    return assignment, curr_err


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
    assignment_base = np.empty(I, dtype=int)
    assignment_base[order[:n_shared]] = SHARED
    remaining = order[n_shared:]

    t0 = time.time()
    best_err_final = 1e9
    best_assign_final = None

    for seed in range(10):
        rng = np.random.default_rng(seed * 7)
        sq_vecs = (act[:, remaining] ** 2).T
        sq_vecs = sq_vecs / (np.linalg.norm(sq_vecs, axis=1, keepdims=True) + 1e-12)
        labels = _kmeans(sq_vecs, n_experts, rng)
        assign = assignment_base.copy()
        for ch, lab in zip(remaining, labels):
            assign[ch] = int(lab)

        err_seed = oracle_topk_error(activations=act, down=down, assignment=assign, top_k=2)
        counts_seed = np.bincount(assign[assign != SHARED], minlength=n_experts)
        violations = max(0, counts_seed.max() - max_size)
        print(f'  seed {seed}: err={err_seed:.6f}, max_size={counts_seed.max()}, violations={violations}')

        # Only repair if already close to balance (max_size ≤ 450)
        if counts_seed.max() <= 450:
            assign_r, err_r = oracle_repair_fast(act, down, assign.copy(), n_experts, max_size, remaining, top_n=30)
            err_check = oracle_topk_error(activations=act, down=down, assignment=assign_r, top_k=2)
            # Run sweep
            for sw in range(30):
                n_moves, assign_r, proxy = sweep_fast(act, assign_r, n_experts, 2, seed*100+sw, max_size)
                if n_moves == 0: break
            err_final = oracle_topk_error(activations=act, down=down, assignment=assign_r, top_k=2)
            counts_final = np.bincount(assign_r[assign_r != SHARED], minlength=n_experts)
            print(f'    -> after repair+sweep: err={err_final:.6f}, sizes={counts_final.tolist()}')
            if err_final < best_err_final:
                best_err_final = err_final
                best_assign_final = assign_r.copy()

        if time.time() - t0 > 120:
            break

    if best_assign_final is not None:
        err = oracle_topk_error(activations=act, down=down, assignment=best_assign_final, top_k=2)
        counts = np.bincount(best_assign_final[best_assign_final != SHARED], minlength=n_experts)
        print(f'  Best with oracle repair: err={err:.6f}, sizes={counts.tolist()}, time={time.time()-t0:.1f}s')

print('\nDone.')
