"""Debug sq_cosine seeding: understand why it helps and if we can make it valid."""
import sys, time
sys.path.insert(0, 'src')
import numpy as np
from moeforge.grouping import oracle_topk_error

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


def _sweep(activations, assignment, n_experts, top_k, rng_seed, max_size):
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
            if dst == src:
                continue
            if exp_sz[dst] >= max_size:
                continue
            if exp_sz[src] <= 1:
                continue
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


def _forced_balance_sweep(activations, assignment, n_experts, top_k, rng_seed, max_size):
    """Force balance: channels in over-full experts MUST move out (best valid destination)."""
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
    rng = np.random.default_rng(rng_seed)
    n_moves = 0
    cur_cov = np.sort(exp_sq, axis=0)[-top_k:, :].sum()

    # First pass: force balance - move channels out of over-full experts
    while exp_sz.max() > max_size:
        overfull = np.where(exp_sz > max_size)[0]
        for e in overfull:
            # Find channels in this expert and move the least valuable one out
            chs_in_e = non_shared[assignment[non_shared] == e]
            # Sort by contribution to proxy (move least contributing first)
            sq_sum = act_sq[:, chs_in_e].sum(axis=0)  # energy per channel
            # Order by energy ascending (move lowest energy first)
            sort_idx = np.argsort(sq_sum)
            for ch_idx in sort_idx:
                i = chs_in_e[ch_idx]
                sq_i = act_sq[:, i]
                new_sq_src = exp_sq[e] - sq_i
                best_dst = -1
                best_loss = float('inf')
                for dst in range(n_experts):
                    if dst == e:
                        continue
                    if exp_sz[dst] >= max_size:
                        continue
                    trial = exp_sq.copy()
                    trial[e] = new_sq_src
                    trial[dst] = exp_sq[dst] + sq_i
                    cov = np.sort(trial, axis=0)[-top_k:, :].sum()
                    loss = cur_cov - cov
                    if loss < best_loss:
                        best_loss = loss
                        best_dst = dst
                if best_dst >= 0:
                    exp_sq[e] = new_sq_src
                    exp_sq[best_dst] = exp_sq[best_dst] + sq_i
                    exp_sz[e] -= 1
                    exp_sz[best_dst] += 1
                    cur_cov -= best_loss
                    assignment[i] = best_dst
                    n_moves += 1
                    break  # Move one at a time, re-check which experts are overfull

    return n_moves, assignment, cur_cov


d3 = np.load('examples/grouping-search/layer3.npz')

act = d3['activations'].astype(np.float64)
imp = d3['importance'].astype(np.float64)
down = d3['down'].astype(np.float64)

n_experts = 8
shared_ratio = 0.125
T, I = act.shape
n_shared = int(round(shared_ratio * I))
n_routed = I - n_shared
max_size = int(2.0 * n_routed / n_experts)

print(f'T={T}, I={I}, n_shared={n_shared}, n_routed={n_routed}, max_size={max_size}')

# Try sq_cosine seed + forced balance repair
rng = np.random.default_rng(0)
sub_rng = np.random.default_rng(rng.integers(0, 2**31))

# sq_cosine seed
order_imp = np.argsort(-imp)
n_shared2 = int(round(shared_ratio * I))
assignment = np.empty(I, dtype=int)
assignment[order_imp[:n_shared2]] = SHARED
remaining = order_imp[n_shared2:]
vecs = (act[:, remaining] ** 2).T
vecs = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12)
labels = _kmeans(vecs, n_experts, sub_rng)
for ch, lab in zip(remaining, labels):
    assignment[ch] = int(lab)

unique, counts = np.unique(assignment[assignment != SHARED], return_counts=True)
print(f'After sq_cosine seed: counts={counts}')

# Force balance repair
t0 = time.time()
n_mv, assignment, proxy = _forced_balance_sweep(act, assignment, n_experts, 2, 0, max_size)
dt = time.time() - t0
unique, counts = np.unique(assignment[assignment != SHARED], return_counts=True)
print(f'After forced balance: n_moves={n_mv}, proxy={proxy:.2f}, counts={counts}, dt={dt:.1f}s')

# Then normal sweep
for sw in range(60):
    n_moves, assignment, proxy = _sweep(act, assignment, n_experts, 2, sw, max_size)
    if n_moves == 0:
        break
unique, counts = np.unique(assignment[assignment != SHARED], return_counts=True)
print(f'After sweep: proxy={proxy:.2f}, counts={counts}')

err = oracle_topk_error(activations=act, down=down, assignment=assignment, top_k=2)
print(f'Oracle error: {err:.6f}')
