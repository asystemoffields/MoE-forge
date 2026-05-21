"""Score a candidate top-k router against the oracle-selection floor on a FIXED carve.

The carve grouping is held constant (a good `balanced_grouping`); the only thing that
varies is the *router* -- the rule that, per token, picks which top-k experts fire. A
real router sees only the FFN input hidden state `h` (cheaply); it does NOT get to see
each expert's actual output (computing that is the dense work carve avoids). So this
evaluator hands the candidate `h` plus the fixed expert weights/stats, asks it to score
the experts, selects top-k by that score, and reports the SAME relative reconstruction
error that `moeforge.grouping.oracle_topk_error` uses -- making the number directly
comparable to the oracle floor (perfect selection by true contribution norm) and to every
grouping-search result.

A candidate file defines TWO functions:

    def build_router(ctx, n_experts, top_k, rng) -> dict[str, np.ndarray]:
        # OFFLINE. May study fixed weights + a calibration token split (even fit
        # hidden->energy in closed form). Returns a small `state` dict of numpy arrays.
        #   ctx["calib_hidden"]:      [Tc, H]  calibration FFN input hidden states
        #   ctx["calib_activations"]: [Tc, I]  calibration gated activations a=silu(h@gate^T)*(h@up^T)
        #   ctx["gate"]: [I, H]   ctx["up"]: [I, H]   ctx["down"]: [H, I]
        #   ctx["assignment"]: [I] ints (SHARED=-1 = always-active, else expert id 0..E-1)
        #   ctx["importance"]: [I] per-channel mean |activation|
        # state total element count must be <= STATE_BUDGET (= mult * n_experts * H);
        # this forbids smuggling the full gate matrix in to recompute activations.

    def route(hidden, state, n_experts, top_k) -> np.ndarray:  # [T, n_experts] scores
        # PER TOKEN. Sees ONLY eval hidden states + the state build_router returned.
        # Higher score => more likely selected; the evaluator takes top-k by score.

Lower mean_error is better. Usage:
  python examples/router-search/eval_router.py --candidate cand.py \
      --layers examples/grouping-search/layer3.npz examples/grouping-search/layer9.npz
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np

from moeforge.grouping import (
    SHARED,
    balanced_grouping,
    channel_importance,
    intermediate_activations,
    magnitude_grouping,
)


def load_candidate(path: Path):
    spec = importlib.util.spec_from_file_location("router_candidate", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for name in ("build_router", "route"):
        if not hasattr(module, name):
            raise ValueError(f"candidate must define {name}(...)")
    return module.build_router, module.route


def state_float_count(state) -> int:
    """Total numpy elements held in a state dict. Scalars are free; anything that is not a
    dict of arrays/scalars is rejected (so candidates cannot hide tensors in opaque objects)."""
    if not isinstance(state, dict):
        raise ValueError("build_router must return a dict[str, np.ndarray]")
    total = 0
    for key, value in state.items():
        if np.isscalar(value):
            continue
        array = np.asarray(value)
        if array.dtype == object:
            raise ValueError(f"state['{key}'] is not a numeric array")
        total += array.size
    return total


def selection_error(
    dense: np.ndarray,
    base: np.ndarray,
    contributions: list[np.ndarray],
    score: np.ndarray,
    top_k: int,
) -> float:
    """Mean over tokens of ||dense - (base + top-k-by-score experts)|| / ||dense||.

    Identical formula to moeforge.grouping.oracle_topk_error; only the per-token expert
    SELECTION differs (here it is argsort over `score` instead of over true norms)."""
    n_experts = len(contributions)
    k = min(int(top_k), n_experts)
    selected = np.argsort(-score, axis=1)[:, :k]
    reconstruction = base.copy()
    for expert in range(n_experts):
        chosen = (selected == expert).any(axis=1)
        if chosen.any():
            reconstruction[chosen] += contributions[expert][chosen]
    error = np.linalg.norm(dense - reconstruction, axis=1) / (np.linalg.norm(dense, axis=1) + 1e-12)
    return float(error.mean())


def build_assignment(activations, importance, args, rng) -> np.ndarray:
    if args.grouping == "magnitude":
        return magnitude_grouping(importance, n_experts=args.experts, shared_ratio=args.shared_ratio)
    transform = "squared" if args.grouping == "balanced-squared" else "abs"
    return balanced_grouping(
        activations, importance, n_experts=args.experts,
        shared_ratio=args.shared_ratio, rng=rng, transform=transform,
    )


def evaluate_layer(path: Path, build_router, route, args) -> dict:
    data = np.load(path)
    if not all(key in data for key in ("hidden", "gate", "up", "down")):
        raise ValueError(
            f"{path.name} lacks router tensors (hidden/gate/up/down). Re-capture with the "
            "updated examples/grouping-search/capture_layer.py."
        )
    hidden = data["hidden"].astype(np.float64)
    gate = data["gate"].astype(np.float64)
    up = data["up"].astype(np.float64)
    down = data["down"].astype(np.float64)
    activations = (
        data["activations"].astype(np.float64)
        if "activations" in data
        else intermediate_activations(hidden, gate, up)
    )
    importance = (
        data["importance"].astype(np.float64)
        if "importance" in data
        else channel_importance(activations)
    )

    # Fixed carve grouping (deterministic; the same for every candidate in a run).
    assignment = np.asarray(build_assignment(activations, importance, args, np.random.default_rng(args.grouping_seed)))

    # Calibration / eval token split: build_router may fit on calib; routing is SCORED on eval.
    token_count = hidden.shape[0]
    cut = max(1, int(round(args.calib_frac * token_count)))
    cut = min(cut, token_count - 1)
    calib = slice(0, cut)
    held = slice(cut, token_count)

    eval_hidden = hidden[held]
    eval_act = activations[held]
    dense = eval_act @ down.T
    shared_mask = assignment == SHARED
    base = eval_act[:, shared_mask] @ down[:, shared_mask].T if shared_mask.any() else np.zeros_like(dense)

    contributions: list[np.ndarray] = []
    norms = np.zeros((eval_hidden.shape[0], args.experts))
    for expert in range(args.experts):
        mask = assignment == expert
        contribution = eval_act[:, mask] @ down[:, mask].T if mask.any() else np.zeros_like(dense)
        contributions.append(contribution)
        norms[:, expert] = np.linalg.norm(contribution, axis=1)

    # Reference points: oracle (perfect selection by true contribution norm) and random.
    oracle = selection_error(dense, base, contributions, norms, args.top_k)
    rng = np.random.default_rng(args.seed)
    random_score = rng.random((eval_hidden.shape[0], args.experts))
    random_error = selection_error(dense, base, contributions, random_score, args.top_k)

    # Candidate router.
    ctx = {
        "calib_hidden": hidden[calib], "calib_activations": activations[calib],
        "gate": gate, "up": up, "down": down,
        "assignment": assignment, "importance": importance,
    }
    state = build_router(ctx, args.experts, args.top_k, np.random.default_rng(args.seed))
    budget = args.state_budget_mult * args.experts * hidden.shape[1]
    floats = state_float_count(state)
    if floats > budget:
        raise ValueError(f"state has {floats} floats > budget {budget} (router must be cheap, not a re-projection)")
    score = np.asarray(route(eval_hidden, state, args.experts, args.top_k), dtype=np.float64)
    if score.shape != (eval_hidden.shape[0], args.experts):
        raise ValueError(f"route returned shape {score.shape}, expected {(eval_hidden.shape[0], args.experts)}")
    if not np.isfinite(score).all():
        raise ValueError("route returned non-finite scores")
    router_error = selection_error(dense, base, contributions, score, args.top_k)

    return {
        "layer": path.stem, "router": router_error, "oracle": oracle, "random": random_error,
        "gap": router_error - oracle, "state_floats": floats, "state_budget": budget,
        "eval_tokens": int(eval_hidden.shape[0]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--layers", type=Path, nargs="+", required=True)
    parser.add_argument("--experts", type=int, default=8)
    parser.add_argument("--shared-ratio", type=float, default=0.125)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--grouping", choices=["balanced-abs", "balanced-squared", "magnitude"], default="balanced-abs")
    parser.add_argument("--grouping-seed", type=int, default=0)
    parser.add_argument("--calib-frac", type=float, default=0.5)
    parser.add_argument("--state-budget-mult", type=float, default=8.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    try:
        build_router, route = load_candidate(args.candidate)
    except Exception as exc:
        print(json.dumps({"error": f"load failed: {exc}"}))
        return

    per_layer = []
    for path in args.layers:
        try:
            per_layer.append(evaluate_layer(path, build_router, route, args))
        except Exception as exc:
            print(json.dumps({"error": f"{path.stem}: {exc}"}))
            return

    mean_error = float(np.mean([item["router"] for item in per_layer]))
    oracle = float(np.mean([item["oracle"] for item in per_layer]))
    random_error = float(np.mean([item["random"] for item in per_layer]))
    print(json.dumps({
        "mean_error": mean_error, "oracle": oracle, "random": random_error,
        "gap": mean_error - oracle, "per_layer": per_layer,
    }, indent=2))


if __name__ == "__main__":
    main()
