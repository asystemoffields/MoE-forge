"""Test oracle-guided balance repair from sq-acts k-means seed.

Key idea: from the sq-acts k-means seed (which has one huge expert), move channels
from the oversized expert to undersized ones, choosing each move to minimize
oracle error increase. This uses `down` (not available in group()), but serves
as an upper bound and reveals which channels are safe to move.
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


def oracle_repair(act, down, assignment, n_experts, max_size, remaining):
    """Balance repair guided by oracle error delta.

    For each channel to move from oversized to undersized expert,
    compute the oracle error change and pick the move with minimum increase.
    """
    T, I = act.shape
    H = down.shape[0]
    n_rem = len(remaining)

    # Build state
    dense = act @ down.T  # [T, H]
    dense_norms = np.linalg.norm(dense, axis=1)  # [T]

    shared_mask = assignment == SHARED
    recon_shared = act[:, shared_mask] @ down[:, shared_mask].T if shared_mask.any() else np.zeros((T, H))

    # Expert contributions
    contribs = {}
    for e in range(n_experts):
        mask = assignment == e
        if mask.any():
            contribs[e] = act[:, mask] @ down[:, mask].T  # [T, H]
        else:
            contribs[e] = np.zeros((T, H))

    # Expert norms
    norms = np.zeros((T, n_experts))
    for e in range(n_experts):
        norms[:, e] = np.linalg.norm(contribs[e], axis=1)

    # Top-2 selection
    sel = np.argsort(-norms, axis=1)[:, :2]  # [T, 2]
    sel_mask = np.zeros((T, n_experts), dtype=bool)
    sel_mask[np.arange(T), sel[:, 0]] = True
    sel_mask[np.arange(T), sel[:, 1]] = True

    # Current reconstruction
    recon = recon_shared.copy()
    for e in range(n_experts):
        tok_mask = sel_mask[:, e]
        if tok_mask.any():
            recon[tok_mask] += contribs[e][tok_mask]

    errors = np.linalg.norm(dense - recon, axis=1) / (dense_norms + 1e-12)
    curr_err = float(errors.mean())

    # Track channel-level stats for analysis
    moved_channels = []
    moved_err_delta = []

    for iteration in range(100000):
        counts = np.bincount(assignment[remaining], minlength=n_experts)
        if counts.max() <= max_size:
            break
        src = int(np.argmax(counts))
        dst = int(np.argmin(counts))
        src_chs = remaining[assignment[remaining] == src]

        # Try each channel in src, pick the one that minimizes oracle error increase
        best_delta = np.inf
        best_ch = None
        best_new_err = None

        for ch in src_chs:
            a_ch = act[:, ch]      # [T]
            d_ch = down[:, ch]     # [H]
            delta = a_ch[:, np.newaxis] * d_ch[np.newaxis, :]  # [T, H]

            # New contributions
            new_c_src = contribs[src] - delta
            new_c_dst = contribs[dst] + delta

            # New norms for src and dst
            new_norm_src = np.linalg.norm(new_c_src, axis=1)
            new_norm_dst = np.linalg.norm(new_c_dst, axis=1)

            # New norms matrix
            new_norms = norms.copy()
            new_norms[:, src] = new_norm_src
            new_norms[:, dst] = new_norm_dst

            # New top-2 selection
            new_sel = np.argsort(-new_norms, axis=1)[:, :2]
            new_sm = np.zeros((T, n_experts), dtype=bool)
            new_sm[np.arange(T), new_sel[:, 0]] = True
            new_sm[np.arange(T), new_sel[:, 1]] = True

            # Affected tokens: where selection changes or src/dst involved
            affected = (sel_mask[:, src] | sel_mask[:, dst] |
                        new_sm[:, src] | new_sm[:, dst])
            aff_idx = np.where(affected)[0]

            if len(aff_idx) == 0:
                new_err = curr_err
            else:
                # Recompute error for affected tokens
                aff_delta = a_ch[aff_idx, np.newaxis] * d_ch[np.newaxis, :]  # [n_aff, H]
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
                best_new_err = new_err

        # Move best channel
        ch = best_ch
        a_ch = act[:, ch]
        d_ch = down[:, ch]
        delta = a_ch[:, np.newaxis] * d_ch[np.newaxis, :]

        new_c_src = contribs[src] - delta
        new_c_dst = contribs[dst] + delta
        contribs[src] = new_c_src
        contribs[dst] = new_c_dst
        norms[:, src] = np.linalg.norm(new_c_src, axis=1)
        norms[:, dst] = np.linalg.norm(new_c_dst, axis=1)

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

        moved_channels.append(ch)
        moved_err_delta.append(best_delta)

        prev_err = curr_err
        curr_err = float(errors.mean())
        assignment[ch] = dst

        if iteration % 20 == 0:
            counts_now = np.bincount(assignment[remaining], minlength=n_experts)
            print(f'    iter {iteration}: err={curr_err:.6f}, max_size={counts_now.max()}, move_delta={best_delta:+.6f}')

    return assignment, curr_err, moved_channels, moved_err_delta


d3 = np.load('examples/grouping-search/layer3.npz')
layers = [('layer3', d3)]

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

    # Use seed 4 which had best oracle error (0.362) for layer3
    rng = np.random.default_rng(4 * 7)
    sq_vecs = (act[:, remaining] ** 2).T
    sq_vecs = sq_vecs / (np.linalg.norm(sq_vecs, axis=1, keepdims=True) + 1e-12)
    labels = _kmeans(sq_vecs, n_experts, rng)
    assign = assignment_base.copy()
    for ch, lab in zip(remaining, labels):
        assign[ch] = int(lab)

    err0 = oracle_topk_error(activations=act, down=down, assignment=assign, top_k=2)
    counts0 = np.bincount(assign[assign != SHARED], minlength=n_experts)
    print(f'  Initial sq-acts seed: err={err0:.6f}, sizes={counts0.tolist()}')

    t0 = time.time()
    assign, final_err, moved_chs, err_deltas = oracle_repair(
        act, down, assign, n_experts, max_size, remaining)
    dt = time.time() - t0

    err = oracle_topk_error(activations=act, down=down, assignment=assign, top_k=2)
    counts = np.bincount(assign[assign != SHARED], minlength=n_experts)
    print(f'\n  After oracle repair ({dt:.1f}s): err={err:.6f}, sizes={counts.tolist()}')
    print(f'  Total err increase from repair: {err - err0:+.6f}')
    print(f'  n_moves: {len(moved_chs)}, avg delta: {np.mean(err_deltas):.6f}')

    # Analyze moved channels: what distinguishes "low delta" from "high delta" channels?
    moved_arr = np.array(moved_chs)
    delta_arr = np.array(err_deltas)
    print(f'\n  Error delta stats: min={delta_arr.min():.6f}, max={delta_arr.max():.6f}, median={np.median(delta_arr):.6f}')

    # What are the properties of low vs high delta channels?
    low_delta_chs = moved_arr[delta_arr < np.percentile(delta_arr, 25)]
    high_delta_chs = moved_arr[delta_arr > np.percentile(delta_arr, 75)]

    for group_name, group_chs in [('low_delta (safe to move)', low_delta_chs), ('high_delta (costly to move)', high_delta_chs)]:
        if len(group_chs) == 0:
            continue
        act_sq_group = act[:, group_chs] ** 2
        ch_mean = act_sq_group.mean(axis=0).mean()
        ch_var = act_sq_group.var(axis=0).mean()
        ch_cv = (act_sq_group.std(axis=0) / (act_sq_group.mean(axis=0) + 1e-12)).mean()
        imp_group = imp[group_chs].mean()
        print(f'  {group_name}: n={len(group_chs)}, mean_sq={ch_mean:.4f}, var_sq={ch_var:.4f}, cv={ch_cv:.4f}, imp={imp_group:.4f}')

print('\nDone.')
