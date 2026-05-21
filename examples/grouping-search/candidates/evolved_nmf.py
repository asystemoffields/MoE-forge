"""gen1_alt: NMF on squared activations with W-weighted balanced assignment.

Approach: fundamentally different from k-means clustering on activation vectors.

Key insight: The oracle picks top-k experts per token by output contribution norm,
which is approximately proportional to the squared activation energy per expert
(the down-projection norms vary by only ~5% across channels in this model, so
||acts_e @ down_e.T||^2 ≈ const * sum_i acts[t,i]^2 for i in expert e).

NMF on squared activations (acts^2 ≈ W @ H) decomposes the energy matrix where:
  - W[T, k]: how much each NMF component accounts for each token's total energy
  - H[k, I]: how each channel loads onto each NMF component

Assignment: the benefit of placing channel i in expert e is
  B[e, i] = sum_t W_norm[t, e] * acts[t, i]^2
i.e. the expected energy channel i contributes to the tokens that expert e "owns".
This directly optimises for which channels should be co-located.

Two NMF seeding strategies are tried in alternation:
  1. Farthest-point init: selects k diverse token prototypes as W-column seeds,
     giving the NMF a head start with well-separated token patterns.
  2. Random Gaussian init: explores different local minima.

Each candidate is validated for balance (all experts used; max <= 2*ideal size)
and the one with best proxy score (mean fraction of energy missed by top-2 experts
per token) is returned.
"""

import numpy as np

SHARED = -1

_N_PAIRS = 25    # number of (farthest-point, random) seed pairs to try
_NMF_MAX_ITERS = 400
_NMF_TOL = 1e-6


def _nmf(V, k, W_init, H_init):
    """Multiplicative-update NMF given initial W, H.  Returns (H, W)."""
    W = W_init.copy()
    H = H_init.copy()
    prev_loss = np.inf
    for it in range(_NMF_MAX_ITERS):
        WtV = W.T @ V
        WtW = W.T @ W
        H *= WtV / (WtW @ H + 1e-10)
        VHt = V @ H.T
        HHt = H @ H.T
        W *= VHt / (W @ HHt + 1e-10)
        if it % 25 == 24:
            with np.errstate(invalid="ignore", divide="ignore"):
                loss = np.linalg.norm(V - W @ H, "fro")
            if abs(prev_loss - loss) / (prev_loss + 1e-10) < _NMF_TOL:
                break
            prev_loss = loss
    return H, W


def _farthest_init(V, k, rng):
    """Farthest-point token-prototype NMF initialisation.

    Select k diverse tokens as anchors (maximally spread in probability-simplex
    space of per-token energy distribution), then initialise W from cosine
    similarities to anchors and H from anchor rows of V.
    """
    T, I_r = V.shape
    tok_vecs = V / (V.sum(axis=1, keepdims=True) + 1e-12)   # [T, I_r] normalised

    start = int(rng.integers(T))
    anchors = [start]
    min_dists = ((tok_vecs - tok_vecs[start]) ** 2).sum(axis=1)

    for _ in range(k - 1):
        nxt = int(np.argmax(min_dists))
        anchors.append(nxt)
        d = ((tok_vecs - tok_vecs[nxt]) ** 2).sum(axis=1)
        min_dists = np.minimum(min_dists, d)

    anchor_vecs = tok_vecs[anchors]                          # [k, I_r]
    W_init = np.maximum(tok_vecs @ anchor_vecs.T, 1e-10)    # [T, k]
    H_init = np.maximum(V[anchors] + 0.01, 1e-10)          # [k, I_r]
    return W_init, H_init


def _random_init(V, k, rng):
    """Standard random Gaussian NMF initialisation."""
    T, I_r = V.shape
    W_init = np.abs(rng.standard_normal((T, k))) + 0.1
    H_init = np.abs(rng.standard_normal((k, I_r))) + 0.1
    return W_init, H_init


def _balanced_assign(benefit, n_experts, max_size):
    """Greedy balanced assignment from benefit matrix [n_experts, I_r].

    Processes channels in order of decreasing margin between best and second-best
    expert (most decisive first), assigning each channel to its highest-benefit
    expert that still has room.
    """
    k, I_r = benefit.shape
    sorted_b = np.sort(benefit, axis=0)[::-1]
    margin = sorted_b[0] - sorted_b[1]
    processing_order = np.argsort(-margin)
    prefs = np.argsort(-benefit, axis=0).T             # [I_r, k]

    labels = np.empty(I_r, dtype=int)
    counts = np.zeros(n_experts, dtype=int)

    for ch in processing_order:
        for exp in prefs[ch]:
            if counts[exp] < max_size:
                labels[ch] = exp
                counts[exp] += 1
                break
        else:
            labels[ch] = int(np.argmin(counts))
            counts[labels[ch]] += 1

    return labels


def _proxy(sq_acts_r, labels, n_experts, top_k=2):
    """Fraction of per-token squared-activation power NOT captured by top-k experts."""
    T = sq_acts_r.shape[0]
    ep = np.zeros((n_experts, T))
    for e in range(n_experts):
        mask = labels == e
        if mask.any():
            ep[e] = sq_acts_r[:, mask].sum(axis=1)
    total = ep.sum(axis=0) + 1e-12
    topk_power = np.sort(ep, axis=0)[-top_k:, :].sum(axis=0)
    return float((1.0 - topk_power / total).mean())


def group(ctx, n_experts, shared_ratio, rng):
    importance = ctx["importance"]
    activations = ctx["activations"]
    top_k = 2  # scoring regime

    I = importance.shape[0]
    n_shared = int(round(shared_ratio * I))

    # Shared channels: top-importance (highest mean |gated activation|)
    order = np.argsort(-importance)
    assignment = np.empty(I, dtype=int)
    assignment[order[:n_shared]] = SHARED
    remaining = order[n_shared:]
    I_r = len(remaining)

    acts_r = activations[:, remaining]       # [T, I_r]
    sq_acts_r = acts_r ** 2                  # [T, I_r] non-negative NMF input

    # Balance constraint: max expert size = floor(2 * ideal)
    ideal = I_r / n_experts
    max_size = int(2 * ideal)

    best_proxy = np.inf
    best_labels = None

    for pair in range(_N_PAIRS):
        pair_rng = np.random.default_rng(int(rng.integers(2**31)))

        for init_fn in (_farthest_init, _random_init):
            seed_rng = np.random.default_rng(int(pair_rng.integers(2**31)))
            W_init, H_init = init_fn(sq_acts_r, n_experts, seed_rng)
            H, W = _nmf(sq_acts_r, n_experts, W_init, H_init)

            # Build W-weighted benefit matrix.
            # Use top-2 token assignment: each token contributes its energy only
            # to its two most-active NMF components (weighted by their W values).
            # This sharpens the expert-token alignment compared to a soft average.
            T_ = W.shape[0]
            W_norm = W / (W.sum(axis=1, keepdims=True) + 1e-10)   # [T, k]
            top2_idx = np.argsort(-W_norm, axis=1)[:, :2]          # [T, 2]
            W_top2 = np.zeros_like(W_norm)
            for t in range(T_):
                idx = top2_idx[t]
                w = W_norm[t, idx]
                W_top2[t, idx] = w / (w.sum() + 1e-10)
            benefit = W_top2.T @ sq_acts_r                          # [k, I_r]

            labels = _balanced_assign(benefit, n_experts, max_size)

            # Validate balance
            counts = np.bincount(labels, minlength=n_experts)
            if (counts == 0).any() or counts.max() > max_size:
                continue

            prx = _proxy(sq_acts_r, labels, n_experts, top_k)
            if prx < best_proxy:
                best_proxy = prx
                best_labels = labels

    # Fallback: round-robin (should never trigger)
    if best_labels is None:
        best_labels = np.arange(I_r, dtype=int) % n_experts

    assignment[remaining] = best_labels
    return assignment
