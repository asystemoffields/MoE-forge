"""gen2_shrunk_calib: output-relevance keys with a heavily-regularised calibration nudge.

Strategy:
- Start with the same fit-free output-weighted key as gen1_outweighted (robust prior).
- Compute the expert energy targets on the calibration split (like gen1_lstsq does).
- Use the fit-free key as a warm-start: regress the RESIDUAL between the (normalised) energy
  target and the fit-free score onto centered hidden, with a very strong ridge.
- Blend only a small, confidence-scaled correction (alpha = 0.15 * mean R^2) into the key.
- On tiny/noisy fixtures the correction is nearly zero (ridge + low R^2 kill it), so we
  degrade to gen1_outweighted. On real data with hundreds of tokens the correction adds value.

State: blended keys [E, H] + h_mean [H] (2 small arrays, well within 8*E*H).
"""

import numpy as np

SHARED = -1


def build_router(ctx, n_experts, top_k, rng):
    gate = ctx["gate"]
    down = ctx["down"]
    assignment = ctx["assignment"]
    importance = ctx["importance"]
    calib_hidden = ctx["calib_hidden"]       # [Tc, H]
    calib_activations = ctx["calib_activations"]  # [Tc, I]

    hidden_dim = gate.shape[1]
    Tc = calib_hidden.shape[0]

    # --- Step 1: build the fit-free output-weighted keys (gen1_outweighted) ---
    down_norm = np.linalg.norm(down, axis=0)  # [I] per-channel output relevance
    keys = np.zeros((n_experts, hidden_dim))
    for expert in range(n_experts):
        mask = assignment == expert
        if not mask.any():
            continue
        weight = importance[mask] * down_norm[mask]
        key = (gate[mask] * weight[:, None]).sum(axis=0) / (weight.sum() + 1e-12)
        keys[expert] = key / (np.linalg.norm(key) + 1e-12)

    # Fit-free score on calibration tokens: [Tc, E]
    fitfree_scores = calib_hidden @ keys.T  # [Tc, E]

    # --- Step 2: calibration energy targets ---
    targets = np.zeros((Tc, n_experts))
    for expert in range(n_experts):
        mask = assignment == expert
        if mask.any():
            contribution = calib_activations[:, mask] @ down[:, mask].T
            targets[:, expert] = np.linalg.norm(contribution, axis=1)

    # Normalise targets to make ridge scale-invariant across layers.
    target_scale = targets.std(axis=0, keepdims=True) + 1e-12  # [1, E]
    targets_normed = targets / target_scale

    # --- Step 3: heavily-ridged regression for a RESIDUAL correction ---
    residual_targets = targets_normed - fitfree_scores  # [Tc, E]

    h_mean = calib_hidden.mean(axis=0, keepdims=True)  # [1, H]
    H_centered = calib_hidden - h_mean                  # [Tc, H]

    ridge_strength = float(Tc)   # strong: one unit of variance per token
    gram = H_centered.T @ H_centered + ridge_strength * np.eye(hidden_dim)  # [H, H]

    rhs = H_centered.T @ residual_targets  # [H, E]
    delta_keys_T = np.linalg.solve(gram, rhs)  # [H, E]
    delta_keys = delta_keys_T.T  # [E, H]

    # --- Step 4: blend --- small alpha so correction can only help, not hurt much.
    pred_residual = H_centered @ delta_keys_T  # [Tc, E]
    ss_res = np.sum((residual_targets - pred_residual) ** 2, axis=0)
    ss_tot = np.sum(residual_targets ** 2, axis=0) + 1e-12
    r2 = np.clip(1.0 - ss_res / ss_tot, 0.0, 1.0)  # [E]
    mean_r2 = float(r2.mean())

    alpha = 0.15 * mean_r2

    blended_keys = keys + alpha * delta_keys  # [E, H]
    norms = np.linalg.norm(blended_keys, axis=1, keepdims=True) + 1e-12
    blended_keys = blended_keys / norms

    return {
        "keys": blended_keys,      # [E, H]
        "h_mean": h_mean[0],       # [H]
        "alpha": alpha,            # scalar (free)
        "mean_r2": mean_r2,        # scalar (free, for debugging)
    }


def route(hidden, state, n_experts, top_k):
    keys = state["keys"]           # [E, H]
    scores = hidden @ keys.T       # [T, E]
    return scores
