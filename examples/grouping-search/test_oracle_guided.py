"""Test oracle-guided refinement approach."""
import numpy as np
import time
import sys
sys.path.insert(0, 'src')
from moeforge.grouping import oracle_topk_error, SHARED


def run_oracle_guided(acts, down, imp, n_experts=8, shared_ratio=0.125, rng=None, n_tries=50):
    I = acts.shape[1]
    T = acts.shape[0]
    H = down.shape[0]
    channel_norms = np.linalg.norm(down, axis=0)
    n_shared = int(round(shared_ratio * I))
    order = np.argsort(-imp)
    remaining = order[n_shared:]
    I_r = len(remaining)

    ch_weights = np.abs(acts[:, remaining]) * channel_norms[remaining][None, :]
    ch_w_norms = np.linalg.norm(ch_weights, axis=0, keepdims=True) + 1e-12
    ch_w_norm = (ch_weights / ch_w_norms).T  # [I_r, T]

    if rng is None:
        rng = np.random.default_rng(0)

    best_labels = None
    best_err = np.inf

    for trial in range(n_tries):
        first_anchor = int(rng.integers(I_r))
        anchors = [first_anchor]
        min_sims = ch_w_norm @ ch_w_norm[first_anchor]

        for _ in range(n_experts - 1):
            next_anchor = int(np.argmin(min_sims))
            anchors.append(next_anchor)
            new_sims = ch_w_norm @ ch_w_norm[next_anchor]
            min_sims = np.maximum(min_sims, new_sims)

        anchor_vecs = ch_w_norm[anchors]
        sims_to_anchors = ch_w_norm @ anchor_vecs.T
        labels = sims_to_anchors.argmax(axis=1)

        # Oracle-guided refinement pass
        assignment = np.full(I, SHARED, dtype=int)
        assignment[remaining] = labels

        expert_contribs = np.zeros((n_experts, T, H))
        for e in range(n_experts):
            mask = assignment == e
            if mask.any():
                expert_contribs[e] = acts[:, mask] @ down[:, mask].T

        expert_norms = np.linalg.norm(expert_contribs, axis=2)  # [E, T]

        # For each token, which 2 experts are selected by oracle?
        selected = np.argsort(-expert_norms, axis=0)[:2, :]  # [2, T]
        expert_tokens = np.zeros((n_experts, T), dtype=bool)
        for k in range(2):
            expert_tokens[selected[k], np.arange(T)] = True

        # Benefit of channel i to expert e = sum of ch_weights[t, i] for tokens where e selected
        benefit = expert_tokens.astype(float) @ ch_weights  # [E, T] @ [T, I_r] = [E, I_r]
        new_labels = benefit.argmax(axis=0)  # [I_r]

        # Evaluate both anchor and refined assignments
        new_assignment = np.full(I, SHARED, dtype=int)
        new_assignment[remaining] = new_labels
        err_new = oracle_topk_error(activations=acts, down=down, assignment=new_assignment, top_k=2)

        orig_assignment = np.full(I, SHARED, dtype=int)
        orig_assignment[remaining] = labels
        err_orig = oracle_topk_error(activations=acts, down=down, assignment=orig_assignment, top_k=2)

        if err_new < err_orig:
            trial_err = err_new
            trial_labels = new_labels
        else:
            trial_err = err_orig
            trial_labels = labels

        if trial_err < best_err:
            best_err = trial_err
            best_labels = trial_labels.copy()

    assignment = np.full(I, SHARED, dtype=int)
    assignment[remaining] = best_labels
    return assignment, best_err


for layer in ['examples/grouping-search/layer3.npz', 'examples/grouping-search/layer9.npz']:
    data = np.load(layer)
    acts = data['activations'].astype(np.float64)
    imp = data['importance'].astype(np.float64)
    down = data['down'].astype(np.float64)

    rng = np.random.default_rng(0)
    t0 = time.time()
    assignment, err = run_oracle_guided(acts, down, imp, rng=rng, n_tries=50)
    t1 = time.time()
    sizes = np.bincount(assignment[assignment >= 0], minlength=8)
    print(f'{layer}: error={err:.4f}, time={t1-t0:.2f}s, sizes={sizes}')
