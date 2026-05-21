"""Test spectral/PCA-based grouping approaches."""
import sys, time
sys.path.insert(0, 'src')
import numpy as np
from moeforge.grouping import oracle_topk_error, SHARED

SHARED = -1


def sweep_topk(activations, assignment, n_experts, top_k, rng_seed, max_size):
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


def make_seed_pca(activations, importance, n_experts, shared_ratio, rng, max_size):
    """PCA of sq-activations, k-means on top-8 PCs."""
    I = importance.shape[0]
    n_shared = int(round(shared_ratio * I))
    order = np.argsort(-importance)
    assignment = np.empty(I, dtype=int)
    assignment[order[:n_shared]] = SHARED
    remaining = order[n_shared:]

    # Use squared activations for the representation
    sq_vecs = (activations[:, remaining] ** 2).T  # [n_rem, T]

    # Normalize each channel vector
    norms = np.linalg.norm(sq_vecs, axis=1, keepdims=True) + 1e-12
    sq_norm = sq_vecs / norms

    # PCA: project to top-K components
    K = min(32, sq_norm.shape[0], sq_norm.shape[1])
    # SVD of the data matrix
    U, s, Vt = np.linalg.svd(sq_norm, full_matrices=False)
    # Use top-K components
    features = U[:, :K] * s[np.newaxis, :K]  # [n_rem, K]

    # K-means on PCA features
    n_rem = len(remaining)
    centers = features[rng.choice(n_rem, n_experts, replace=False)].copy()
    labels = np.zeros(n_rem, dtype=int)
    for _ in range(25):
        distances = ((features[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        labels = distances.argmin(axis=1)
        for c in range(n_experts):
            members = features[labels == c]
            if len(members):
                centers[c] = members.mean(axis=0)

    for ch, lab in zip(remaining, labels):
        assignment[ch] = int(lab)

    # Balance repair
    ch_energy = dict(zip(remaining.tolist(), (activations[:, remaining]**2).sum(axis=0).tolist()))
    for _ in range(100000):
        counts = np.bincount(assignment[remaining], minlength=n_experts)
        if counts.max() <= max_size:
            break
        src = int(np.argmax(counts))
        src_chs = remaining[assignment[remaining] == src]
        energies = np.array([ch_energy[c] for c in src_chs])
        lowest_idx = int(np.argmin(energies))
        ch = src_chs[lowest_idx]
        dst = int(np.argmin(counts))
        assignment[ch] = dst

    return assignment


def make_seed_svd_channels(activations, importance, n_experts, shared_ratio, rng):
    """Group channels by their top SVD component (which token pattern they respond to).

    Idea: channels that respond to different token patterns should be in different
    experts. We assign channels based on their dominant activation pattern.
    """
    I = importance.shape[0]
    n_shared = int(round(shared_ratio * I))
    order = np.argsort(-importance)
    assignment = np.empty(I, dtype=int)
    assignment[order[:n_shared]] = SHARED
    remaining = order[n_shared:]

    # For each channel, compute its activation pattern (L2-normed)
    act_rem = activations[:, remaining]  # [T, n_rem]
    norms = np.linalg.norm(act_rem, axis=0, keepdims=True) + 1e-12  # [1, n_rem]
    act_norm = act_rem / norms  # [T, n_rem]

    # SVD of activation matrix
    K = min(n_experts * 4, act_norm.shape[0], act_norm.shape[1])
    U, s, Vt = np.linalg.svd(act_norm, full_matrices=False)
    # Channel embeddings: right singular vectors
    ch_embed = Vt[:K].T  # [n_rem, K]

    # K-means on channel embeddings
    n_rem = len(remaining)
    centers = ch_embed[rng.choice(n_rem, n_experts, replace=False)].copy()
    labels = np.zeros(n_rem, dtype=int)
    for _ in range(25):
        distances = ((ch_embed[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        labels = distances.argmin(axis=1)
        for c in range(n_experts):
            members = ch_embed[labels == c]
            if len(members):
                centers[c] = members.mean(axis=0)

    for ch, lab in zip(remaining, labels):
        assignment[ch] = int(lab)
    return assignment


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
    n_routed = I - int(round(shared_ratio * I))
    max_size = int(2.0 * n_routed / n_experts)
    print(f'\n=== {layer_name} ===')

    # Approach: PCA seed + balanced top-2 sweep
    t0 = time.time()
    best_ps = -1
    best_assign = None
    for try_idx in range(5):
        if time.time() - t0 > 22: break
        rng = np.random.default_rng(try_idx * 37 + 111)
        assign = make_seed_pca(act, imp, n_experts, shared_ratio, rng, max_size)
        err_seed = oracle_topk_error(activations=act, down=down, assignment=assign, top_k=2)
        counts_seed = np.bincount(assign[assign != SHARED], minlength=n_experts)
        print(f'  PCA seed {try_idx}: err={err_seed:.6f}, sizes={counts_seed.tolist()}')
        for sweep in range(60):
            n_moves, assign, proxy = sweep_topk(act, assign, n_experts, 2, try_idx*100+sweep, max_size)
            if n_moves == 0: break
        if proxy > best_ps:
            best_ps = proxy
            best_assign = assign.copy()
    err = oracle_topk_error(activations=act, down=down, assignment=best_assign, top_k=2)
    counts = np.bincount(best_assign[best_assign != SHARED], minlength=n_experts)
    print(f'  PCA (best of 5): err={err:.6f}, sizes={counts.tolist()}, time={time.time()-t0:.1f}s')

print('\nDone.')
