"""gen2_actkey: empirical + analytical blended key with output-magnitude scale.

Two improvements over gen1_outweighted:
1. Key direction: blend the analytical (gate+up weighted) direction with an empirical
   direction from calibration (weighted mean of calib hidden states by expert's
   output-relevant activation on calibration tokens). The empirical key captures the actual
   silu+up nonlinearity implicitly, without any fit.
2. Score scaling: multiply dot-product scores by each expert's mean output norm on
   calibration tokens (normalized across experts). This biases selection toward experts
   that historically produce larger contributions, moving routing from pure
   direction-matching toward magnitude prediction -- closer to the oracle, which ranks by
   true output norm.
"""

import numpy as np

SHARED = -1


def build_router(ctx, n_experts, top_k, rng):
    gate = ctx["gate"]               # [I, H]
    up = ctx["up"]                   # [I, H]
    down = ctx["down"]               # [H, I]
    assignment = ctx["assignment"]   # [I]
    importance = ctx["importance"]   # [I]
    calib_hidden = ctx["calib_hidden"]       # [Tc, H]
    calib_act = ctx["calib_activations"]     # [Tc, I]

    hidden_dim = gate.shape[1]
    Tc = calib_hidden.shape[0]

    down_norm = np.linalg.norm(down, axis=0)  # [I] per-channel output relevance

    # How much to blend empirical vs analytical: more calibration tokens -> more empirical
    calib_trust = np.sqrt(float(Tc)) / (np.sqrt(float(Tc)) + 4.0)

    keys = np.zeros((n_experts, hidden_dim))
    scales = np.ones(n_experts)

    for expert in range(n_experts):
        mask = assignment == expert
        if not mask.any():
            continue

        # Per-channel weight: importance * output-magnitude (same weighting as gen1_outweighted)
        ch_weight = importance[mask] * down_norm[mask]
        ch_sum = ch_weight.sum()
        if ch_sum < 1e-12:
            ch_weight = np.ones(mask.sum(), dtype=np.float64) / max(mask.sum(), 1)
        else:
            ch_weight = ch_weight / ch_sum

        # Analytical key: blend gate and up rows (both shape the gated activation)
        gate_key = (gate[mask] * ch_weight[:, None]).sum(axis=0)  # [H]
        up_key = (up[mask] * ch_weight[:, None]).sum(axis=0)      # [H]
        analytic_key = gate_key + up_key  # additive blend
        analytic_norm = np.linalg.norm(analytic_key)
        if analytic_norm > 1e-12:
            analytic_key = analytic_key / analytic_norm

        # Empirical key: calibration hidden centroid weighted by expert activation strength
        token_scores = calib_act[:, mask] @ ch_weight  # [Tc] weighted activation per token
        token_scores = np.maximum(token_scores, 0.0)   # only positive contributions matter
        score_sum = token_scores.sum()
        if score_sum > 1e-12:
            token_weights = token_scores / score_sum
            empirical_key = calib_hidden.T @ token_weights  # [H]
            emp_norm = np.linalg.norm(empirical_key)
            if emp_norm > 1e-12:
                empirical_key = empirical_key / emp_norm
            else:
                empirical_key = analytic_key.copy()
        else:
            empirical_key = analytic_key.copy()

        # Blend analytical and empirical
        key = calib_trust * empirical_key + (1.0 - calib_trust) * analytic_key
        key_norm = np.linalg.norm(key)
        keys[expert] = key / (key_norm + 1e-12)

        # Scale estimate: mean output-contribution norm of this expert on calibration tokens
        expert_contrib = calib_act[:, mask] @ down[:, mask].T  # [Tc, H]
        expert_norms = np.linalg.norm(expert_contrib, axis=1)  # [Tc]
        scales[expert] = float(expert_norms.mean()) + 1e-12

    # Normalize scales to unit mean (relative ranking across experts is what matters)
    scale_mean = scales.mean()
    if scale_mean > 1e-12:
        scales = scales / scale_mean

    return {"keys": keys, "scales": scales}


def route(hidden, state, n_experts, top_k):
    # Direction score scaled by expert output relevance
    scores = hidden @ state["keys"].T        # [T, n_experts]
    return scores * state["scales"][None, :]  # bias toward high-output experts
