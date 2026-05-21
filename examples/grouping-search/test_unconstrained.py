"""Analyze unconstrained sq-acts k-means to understand channel structure."""
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
    ideal = n_routed / n_experts
    print(f'\n=== {layer_name} === (I={I}, n_routed={n_routed}, ideal={ideal:.0f}, max_size={max_size})')

    order = np.argsort(-imp)
    assignment = np.empty(I, dtype=int)
    assignment[order[:n_shared]] = SHARED
    remaining = order[n_shared:]

    # Unconstrained sq-acts k-means
    sq_vecs = (act[:, remaining] ** 2).T
    sq_vecs = sq_vecs / (np.linalg.norm(sq_vecs, axis=1, keepdims=True) + 1e-12)

    best_err = 1e9
    best_sizes = None
    best_assign = None

    for seed in range(10):
        rng = np.random.default_rng(seed * 7)
        labels = _kmeans(sq_vecs, n_experts, rng)
        assign = assignment.copy()
        for ch, lab in zip(remaining, labels):
            assign[ch] = int(lab)
        err = oracle_topk_error(activations=act, down=down, assignment=assign, top_k=2)
        counts = np.bincount(assign[assign != SHARED], minlength=n_experts)
        # Check if valid (balance)
        valid = counts.max() <= max_size
        print(f'  seed {seed}: err={err:.6f}, sizes={counts.tolist()}, valid={valid}')
        if err < best_err:
            best_err = err
            best_sizes = counts
            best_assign = assign.copy()

    print(f'  Best unconstrained: err={best_err:.6f}, sizes={best_sizes.tolist()}')

    # Check: what's the proxy score for these?
    act_sq = act ** 2
    exp_sq = np.zeros((n_experts, act.shape[0]))
    for e in range(n_experts):
        mask = best_assign == e
        if mask.any():
            exp_sq[e] = act_sq[:, mask].sum(axis=1)
    proxy = np.sort(exp_sq, axis=0)[-2:, :].sum()
    print(f'  Proxy score: {proxy:.2f}')

    # Compare: what does the balanced gen1_factor get?
    # Load gen1 result
    print(f'\n  For reference: gen1_factor achieved err~0.42 with 4x336 + 4x1-2 sizes')
    # Compute proxy for a "4 big experts at 336" structure
    # The proxy should be much higher for 4 big experts

    # Count: how many tokens would a 4x336 structure cover correctly?
    # Let's check the sq-acts energy distribution
    total_sq_per_ch = act_sq[:, remaining].sum(axis=0)  # [n_routed] total sq per channel
    sorted_by_energy = np.argsort(-total_sq_per_ch)
    top_half_chs = sorted_by_energy[:n_routed//2]  # top 50% by energy
    top_half_energy = total_sq_per_ch[top_half_chs].sum()
    total_energy = total_sq_per_ch.sum()
    print(f'  Energy concentration: top 50% channels have {top_half_energy/total_energy*100:.1f}% of total sq-energy')

    top_25_chs = sorted_by_energy[:n_routed//4]
    top_25_energy = total_sq_per_ch[top_25_chs].sum()
    print(f'  Energy concentration: top 25% channels have {top_25_energy/total_energy*100:.1f}% of total sq-energy')

print('\nDone.')
