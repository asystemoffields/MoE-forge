"""gen2_silu_joint

Approximates the SiLU-gated multiplicative activation energy per expert.

The true per-channel activation is  a_i = silu(h @ gate_i) * (h @ up_i).
silu passes only POSITIVE pre-activations, so an expert fires significantly only
when h aligns positively with BOTH its gate rows AND its up rows.

All current candidates score by gate alignment alone; this one captures the joint
gate-AND-up requirement using the product structure of the true activation.

Per expert we build two compact unit-norm keys weighted by output relevance
(importance * ||down_col||):
  gate_key: weighted mean gate direction
  up_key:   weighted mean up direction

Routing score:
  score_e = relu(h . gate_key_e) * relu(h . up_key_e)  +  eps * (h . gate_key_e)

Product is large only when BOTH alignments are positive, closely mirroring
silu(h.gate) * (h.up). The tiny additive gate term breaks ties and prevents
all-zero collapse for tokens where no expert has positive gate alignment.

State budget: 2 * n_experts * H floats (well within 8 * n_experts * H).
"""

import numpy as np

SHARED = -1


def build_router(ctx, n_experts, top_k, rng):
    gate       = ctx["gate"]
    up         = ctx["up"]
    down       = ctx["down"]
    assignment = ctx["assignment"]
    importance = ctx["importance"]

    down_norm     = np.linalg.norm(down, axis=0)
    output_weight = importance * down_norm

    H         = gate.shape[1]
    gate_keys = np.zeros((n_experts, H))
    up_keys   = np.zeros((n_experts, H))

    for expert in range(n_experts):
        mask = assignment == expert
        if not mask.any():
            continue
        w     = output_weight[mask]
        w_sum = w.sum() + 1e-12
        gk                = (gate[mask] * w[:, None]).sum(axis=0) / w_sum
        gate_keys[expert] = gk / (np.linalg.norm(gk) + 1e-12)
        uk              = (up[mask] * w[:, None]).sum(axis=0) / w_sum
        up_keys[expert] = uk / (np.linalg.norm(uk) + 1e-12)

    return {"gate_keys": gate_keys, "up_keys": up_keys}


def route(hidden, state, n_experts, top_k):
    gate_keys   = state["gate_keys"]
    up_keys     = state["up_keys"]
    gate_scores = hidden @ gate_keys.T
    up_scores   = hidden @ up_keys.T
    gate_relu   = np.maximum(gate_scores, 0.0)
    up_relu     = np.maximum(up_scores,   0.0)
    scores      = gate_relu * up_relu + 1e-6 * gate_scores
    return scores
