"""Compare seed quality: what oracle error do seeds achieve without any refinement?"""
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


def _kmeans_balanced(points, k, rng, max_size, iters=30):
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
    strict_max = int(np.ceil(ideal))  # ~168 per expert
    max_size = int(2.0 * ideal)       # 336 per expert
    print(f'\n=== {layer_name} === (ideal={ideal:.0f}, strict_max={strict_max}, max_size={max_size})')

    order = np.argsort(-imp)
    assignment_base = np.empty(I, dtype=int)
    assignment_base[order[:n_shared]] = SHARED
    remaining = order[n_shared:]

    # Standard cosine k-means (unbalanced)
    print('  Standard cosine k-means seeds (no refinement):')
    for seed in range(3):
        rng = np.random.default_rng(seed * 7)
        vecs = act[:, remaining].T
        vecs = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12)
        labels = _kmeans(vecs, n_experts, rng)
        assign = assignment_base.copy()
        for ch, lab in zip(remaining, labels):
            assign[ch] = int(lab)
        err = oracle_topk_error(activations=act, down=down, assignment=assign, top_k=2)
        counts = np.bincount(assign[assign != SHARED], minlength=n_experts)
        print(f'    seed {seed}: err={err:.6f}, sizes={counts.tolist()}')

    # Strictly balanced cosine k-means (max_size=ideal)
    print('  Strictly balanced cosine k-means seeds (no refinement):')
    for seed in range(3):
        rng = np.random.default_rng(seed * 7)
        vecs = act[:, remaining].T
        vecs = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12)
        labels = _kmeans_balanced(vecs, n_experts, rng, strict_max)
        assign = assignment_base.copy()
        for ch, lab in zip(remaining, labels):
            assign[ch] = int(lab)
        counts = np.bincount(assign[assign != SHARED], minlength=n_experts)
        valid = counts.max() <= max_size
        err = oracle_topk_error(activations=act, down=down, assignment=assign, top_k=2)
        print(f'    seed {seed}: err={err:.6f}, sizes={counts.tolist()}, valid={valid}')

    # Sq-acts k-means (unbalanced)
    print('  Sq-acts k-means seeds (no refinement):')
    for seed in range(3):
        rng = np.random.default_rng(seed * 7)
        sq_vecs = (act[:, remaining] ** 2).T
        sq_vecs = sq_vecs / (np.linalg.norm(sq_vecs, axis=1, keepdims=True) + 1e-12)
        labels = _kmeans(sq_vecs, n_experts, rng)
        assign = assignment_base.copy()
        for ch, lab in zip(remaining, labels):
            assign[ch] = int(lab)
        err = oracle_topk_error(activations=act, down=down, assignment=assign, top_k=2)
        counts = np.bincount(assign[assign != SHARED], minlength=n_experts)
        print(f'    seed {seed}: err={err:.6f}, sizes={counts.tolist()}')

print('\nDone.')
