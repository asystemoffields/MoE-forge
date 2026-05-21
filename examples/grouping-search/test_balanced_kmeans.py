"""Test perfectly balanced k-means (all experts ~equal size) vs current approach."""
import sys, time
sys.path.insert(0, 'src')
import numpy as np
from moeforge.grouping import oracle_topk_error, SHARED

SHARED = -1

def _kmeans_balanced(points, k, rng, max_size, iters=30):
    """K-means with hard per-cluster capacity constraint during assignment."""
    n = points.shape[0]
    centers = points[rng.choice(n, k, replace=False)].copy()
    labels = np.zeros(n, dtype=int)

    for _ in range(iters):
        dists = ((points[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        sorted_clusters = np.argsort(dists, axis=1)
        nearest_dist = dists[np.arange(n), sorted_clusters[:, 0]]
        process_order = np.argsort(nearest_dist)
        counts = np.zeros(k, dtype=int)
        new_labels = np.full(n, -1, dtype=int)
        for idx in process_order:
            for rank in range(k):
                c = sorted_clusters[idx, rank]
                if counts[c] < max_size:
                    new_labels[idx] = c
                    counts[c] += 1
                    break
        labels = new_labels
        for c in range(k):
            members = points[labels == c]
            if len(members):
                centers[c] = members.mean(axis=0)

    return labels


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
    ideal = n_routed / n_experts
    # Strict balance: each expert has exactly ceil(ideal) channels
    strict_max = int(np.ceil(ideal))
    max_size = int(2.0 * ideal)  # official limit
    print(f'\n=== {layer_name} === (n_routed={n_routed}, ideal={ideal:.0f}, strict_max={strict_max}, max_size={max_size})')

    order = np.argsort(-imp)
    assignment_base = np.empty(I, dtype=int)
    assignment_base[order[:n_shared]] = SHARED
    remaining = order[n_shared:]

    t0 = time.time()
    best_err = 1e9
    best_assign = None
    for try_idx in range(5):
        rng = np.random.default_rng(try_idx * 37)
        vecs = act[:, remaining].T
        vecs = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12)

        # Strict balance (ceil(ideal) per expert)
        labels = _kmeans_balanced(vecs, n_experts, rng, strict_max)
        assign = assignment_base.copy()
        for ch, lab in zip(remaining, labels):
            assign[ch] = int(lab)
        err = oracle_topk_error(activations=act, down=down, assignment=assign, top_k=2)
        counts = np.bincount(assign[assign != SHARED], minlength=n_experts)
        if try_idx == 0:
            print(f'  Balanced seed 0: err={err:.6f}, sizes={counts.tolist()}')
        if err < best_err:
            best_err = err
            best_assign = assign.copy()

    print(f'  Best strict-balanced seed (5 tries): err={best_err:.6f}, time={time.time()-t0:.2f}s')

    # Now: what if we DON'T refine at all, just use the strict-balanced seed?
    # Compare: standard k-means seed (unbalanced) without refinement
    rng = np.random.default_rng(42)
    vecs = act[:, remaining].T
    vecs = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12)
    from examples.grouping_search.test_balanced_kmeans import _kmeans_balanced  # nope, inline it
    centers = vecs[rng.choice(len(remaining), n_experts, replace=False)].copy()
    labels = np.zeros(len(remaining), dtype=int)
    for _ in range(25):
        dists = ((vecs[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        labels = dists.argmin(axis=1)
        for c in range(n_experts):
            members = vecs[labels == c]
            if len(members):
                centers[c] = members.mean(axis=0)

    assign_std = assignment_base.copy()
    for ch, lab in zip(remaining, labels):
        assign_std[ch] = int(lab)
    err_std = oracle_topk_error(activations=act, down=down, assignment=assign_std, top_k=2)
    counts_std = np.bincount(assign_std[assign_std != SHARED], minlength=n_experts)
    print(f'  Standard k-means seed: err={err_std:.6f}, sizes={counts_std.tolist()}')

print('\nDone.')
