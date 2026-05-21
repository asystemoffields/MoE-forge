"""evolved_energy: the validated winner of the router-evolve search (Sonnet-spawned, gen2).

Scored on real captured SmolLM-135M layers it beats the gate-direction seed and every other
candidate at every operating point tested, and the win holds on a HELD-OUT layer (not metric
gaming):

  setting               oracle  random  evolved_energy  seed_router
  train(L3+L9) 8/top2   0.5545  0.6194  0.5942          0.5958
  held-out L6  8/top2   0.6006  0.6791  0.6426          0.6482
  train(L3+L9) 4/top2   0.4473  0.5041  0.4825          0.4837

(Note: on the synthetic fixture this same rule ranked LAST -- random Gaussian weights have none
of the gate/up/down structure it exploits. Only real layers rank routers; see README.)

--- mechanism ---
The oracle selects experts by ||contribution_e(h)|| where contribution_e = act[:,mask_e] @ down[:,mask_e]^T
and act_i = silu(h@gate_i) * (h@up_i). The product (h@gate_i)*(h@up_i) is bounded above by
((h@gate_i) + (h@up_i))^2 / 4 (AM-GM), and for active channels where both terms are positive this
is a tight approximation. So the squared contribution norm is approximately:
  ||C_e(h)||^2 ~ (1/4) sum_i [ w_i * (h^T (gate_i + up_i))^2 ]  where w_i = importance_i * ||down_col_i||^2
              = h^T M_e h  where M_e = (1/4) sum_i w_i r_i r_i^T,  r_i = gate_i + up_i

M_e is PSD. We compress it to its top K eigenvectors (absorbing sqrt(eigenvalue) into each vector)
and score at route time as sum_k (h @ v_ek)^2 = ||h @ V_e^T||^2 -- a genuine second-order energy
proxy, O(n_experts * K * H) per token. Blended with the output-weighted linear key for stability,
plus a log-mean-energy bias from calibration tokens (a free, non-overfitting prior on expert activity).

State budget: E*(N_EIGVEC+1)*H + 3*E scalars = 6*E*H (for N_EIGVEC=5) < 8*E*H limit.
"""

import numpy as np

SHARED = -1
N_EIGVEC = 5   # eigenvectors per expert; state = E*(N_EIGVEC+1)*H <= 6*E*H < 8*E*H budget
BLEND = 0.2    # weight on the linear (gate-direction) fallback term


def build_router(ctx, n_experts, top_k, rng):
    gate = ctx["gate"]            # [I, H]
    up = ctx["up"]                # [I, H]
    down = ctx["down"]            # [H, I]
    assignment = ctx["assignment"]  # [I]
    importance = ctx["importance"]  # [I]
    I, H = gate.shape

    calib_h = ctx["calib_hidden"]         # [Tc, H]
    calib_act = ctx["calib_activations"]  # [Tc, I]
    Tc = calib_h.shape[0]

    down_norm = np.linalg.norm(down, axis=0)  # [I]

    eigvecs = np.zeros((n_experts, N_EIGVEC, H))  # scaled top eigenvectors of M_e
    lin_keys = np.zeros((n_experts, H))            # linear gate-direction keys
    calib_energy = np.zeros(n_experts)             # mean calib contribution norm per expert

    for expert in range(n_experts):
        mask = assignment == expert
        if not mask.any():
            continue

        imp = importance[mask]    # [C]
        dn = down_norm[mask]      # [C]
        g = gate[mask]            # [C, H]
        u = up[mask]              # [C, H]

        # Combined direction captures both gate and up projections
        r = g + u                 # [C, H]

        # Per-channel weight for second-order energy matrix
        w2 = imp * dn ** 2        # [C]  (importance * ||down_col||^2)

        # M_e = r^T diag(w2) r  (PSD, [H, H])
        M = (r * w2[:, None]).T @ r

        # Extract top N_EIGVEC eigenvectors, scaled by sqrt(eigenvalue)
        k = min(N_EIGVEC, H)
        try:
            eigvals, vecs = np.linalg.eigh(M)  # ascending order
            order = np.argsort(eigvals)[::-1][:k]
            for j, ci in enumerate(order):
                lam = max(eigvals[ci], 0.0)
                eigvecs[expert, j] = vecs[:, ci] * np.sqrt(lam)
        except np.linalg.LinAlgError:
            pass

        # Linear key: importance * down_norm weighted gate (gen1_outweighted formula)
        wl = imp * dn
        key = (g * wl[:, None]).sum(axis=0) / (wl.sum() + 1e-12)
        lin_keys[expert] = key / (np.linalg.norm(key) + 1e-12)

        # Calibration mean energy (used only for a scalar bias -- extremely robust)
        contrib = calib_act[:, mask] @ down[:, mask].T  # [Tc, H]
        calib_energy[expert] = np.linalg.norm(contrib, axis=1).mean()

    # Normalize scores over calibration tokens so both terms are on the same scale
    calib_quad = np.zeros((Tc, n_experts))
    calib_lin = calib_h @ lin_keys.T  # [Tc, E]
    flat_eigvecs = eigvecs.reshape(n_experts * N_EIGVEC, H)
    for expert in range(n_experts):
        proj = calib_h @ eigvecs[expert].T  # [Tc, N_EIGVEC]
        calib_quad[:, expert] = (proj ** 2).sum(axis=1)

    quad_std = calib_quad.std(axis=0) + 1e-12  # [E]
    lin_std = calib_lin.std(axis=0) + 1e-12    # [E]
    energy_bias = np.log(calib_energy + 1e-12)  # [E] log mean energy prior

    return {
        "eigvecs": flat_eigvecs,       # [E*N_EIGVEC, H]
        "lin_keys": lin_keys,           # [E, H]
        "quad_std": quad_std,           # [E]
        "lin_std": lin_std,             # [E]
        "energy_bias": energy_bias,     # [E]
        "n_eigvec": np.array([N_EIGVEC]),
        "blend": np.array([BLEND]),
    }


def route(hidden, state, n_experts, top_k):
    K = int(state["n_eigvec"][0])
    blend = float(state["blend"][0])
    T, H = hidden.shape

    eigvecs = state["eigvecs"].reshape(n_experts, K, H)  # [E, K, H]
    quad_std = state["quad_std"]        # [E]
    lin_std = state["lin_std"]          # [E]
    lin_keys = state["lin_keys"]        # [E, H]
    energy_bias = state["energy_bias"]  # [E]

    # Second-order score: sum_k (h @ v_ek)^2 for each expert
    all_proj = hidden @ eigvecs.reshape(n_experts * K, H).T  # [T, E*K]
    quad_score = (all_proj.reshape(T, n_experts, K) ** 2).sum(axis=2)  # [T, E]
    quad_score = quad_score / quad_std  # normalize to unit std (from calibration)

    # Linear fallback score (gen1_outweighted-style)
    lin_score = (hidden @ lin_keys.T) / lin_std  # [T, E]

    # Combined score + log-energy prior
    return (1.0 - blend) * quad_score + blend * lin_score + energy_bias
