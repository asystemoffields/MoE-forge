"""Check timing of gen4_swap candidate."""
import sys, time
sys.path.insert(0, 'src')
import numpy as np
from moeforge.grouping import oracle_topk_error, SHARED

SHARED = -1

import importlib.util
spec = importlib.util.spec_from_file_location('cand', 'examples/grouping-search/candidates/gen4_swap.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

d3 = np.load('examples/grouping-search/layer3.npz')
act = d3['activations'].astype(np.float64)
imp = d3['importance'].astype(np.float64)
down = d3['down'].astype(np.float64)
ctx = {'activations': act, 'importance': imp}

# Monkey-patch to track restarts
original_group = mod.group

def timed_group(ctx, n_experts, shared_ratio, rng, top_k=2, time_limit=30.0):
    """Wrap group to count restarts."""
    import time as _time
    T, I = ctx['activations'].shape
    n_shared = int(round(shared_ratio * I))
    n_routed = I - n_shared
    max_size = int(2.0 * n_routed / n_experts)
    best_assign = None
    best_proxy = -1.0
    t_start = _time.time()
    trial = 0
    while _time.time() - t_start < time_limit:
        sub_rng = np.random.default_rng(rng.integers(0, 2**31) + trial)
        assignment = mod._make_seed(ctx['activations'], ctx['importance'], n_experts, shared_ratio, sub_rng)
        proxy = 0.0
        for sweep in range(60):
            n_moves, assignment, proxy = mod._sweep(
                ctx['activations'], assignment, n_experts, top_k,
                rng_seed=trial * 100 + sweep, max_size=max_size)
            if n_moves == 0: break
        for sp in range(10):
            n_swaps, assignment, pn = mod._swap_sweep(
                ctx['activations'], assignment, n_experts, top_k,
                rng_seed=trial * 1000 + sp, n_candidates=8)
            if pn > proxy: proxy = pn
            if n_swaps == 0: break
            for sw in range(30):
                n_moves, assignment, p2 = mod._sweep(
                    ctx['activations'], assignment, n_experts, top_k,
                    rng_seed=trial * 10000 + sp * 30 + sw, max_size=max_size)
                if p2 > proxy: proxy = p2
                if n_moves == 0: break
        if proxy > best_proxy:
            best_proxy = proxy
            best_assign = assignment.copy()
        elapsed = _time.time() - t_start
        print(f'  Restart {trial}: proxy={proxy:.2f}, elapsed={elapsed:.2f}s')
        trial += 1
    return best_assign

out = open('examples/grouping-search/timing_results.txt', 'w')

rng = np.random.default_rng(0)
t0 = time.time()
best_assign = timed_group(ctx, 8, 0.125, rng, time_limit=50.0)
dt = time.time() - t0
err = oracle_topk_error(activations=act, down=down, assignment=best_assign, top_k=2)
out.write(f'layer3: err={err:.6f}, time={dt:.1f}s\n')
out.close()
print(f'Final: err={err:.6f}')
