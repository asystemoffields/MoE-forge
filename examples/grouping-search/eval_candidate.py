"""Score a candidate carve-grouping function against the oracle-top-k metric.

A candidate is a Python file defining:

    def group(ctx, n_experts, shared_ratio, rng):
        # ctx["importance"]: float64 [I] per-channel mean |gated activation|
        # ctx["activations"]: float64 [T, I] per-token gated activations
        # return: int np.ndarray [I]; -1 (SHARED) = always-active channel, else expert id 0..n_experts-1
        ...

Lower mean error is better. Usage:
  python examples/grouping-search/eval_candidate.py --candidate cand.py \
      --layers examples/grouping-search/layer3.npz examples/grouping-search/layer9.npz
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np

from moeforge.grouping import SHARED, oracle_topk_error


def load_group(path: Path):
    spec = importlib.util.spec_from_file_location("candidate", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "group"):
        raise ValueError("candidate must define group(ctx, n_experts, shared_ratio, rng)")
    return module.group


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--layers", type=Path, nargs="+", required=True)
    parser.add_argument("--experts", type=int, default=8)
    parser.add_argument("--shared-ratio", type=float, default=0.125)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--max-balance", type=float, default=2.0, help="Reject if any expert > this * ideal size.")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    try:
        group = load_group(args.candidate)
    except Exception as exc:  # candidate import/syntax failure
        print(json.dumps({"error": f"load failed: {exc}"}))
        return

    per_layer = []
    for path in args.layers:
        data = np.load(path)
        activations = data["activations"].astype(np.float64)
        down = data["down"].astype(np.float64)
        ctx = {"importance": data["importance"].astype(np.float64), "activations": activations}
        rng = np.random.default_rng(args.seed)
        try:
            assignment = np.asarray(group(ctx, args.experts, args.shared_ratio, rng))
        except Exception as exc:
            print(json.dumps({"error": f"group() raised on {path.stem}: {exc}"}))
            return
        if assignment.shape != (activations.shape[1],):
            print(json.dumps({"error": f"bad assignment shape {assignment.shape} on {path.stem}"}))
            return
        valid = (assignment == SHARED) | ((assignment >= 0) & (assignment < args.experts))
        if not bool(valid.all()):
            print(json.dumps({"error": f"assignment has out-of-range values on {path.stem}"}))
            return
        # Budget guard: oracle-top-k is trivially 0 if a grouping isn't actually sparse
        # (e.g. all routed channels in one expert, so top-k covers everything). Require all
        # experts used and roughly balanced so top-k drops a real fraction of channels.
        routed = assignment[assignment != SHARED]
        counts = np.bincount(routed, minlength=args.experts) if routed.size else np.zeros(args.experts, dtype=int)
        if int((counts == 0).sum()) > 0:
            print(json.dumps({"error": f"not all {args.experts} experts used on {path.stem}: sizes {counts.tolist()}"}))
            return
        ideal = routed.size / args.experts
        if ideal > 0 and int(counts.max()) > args.max_balance * ideal:
            print(json.dumps({"error": f"unbalanced experts on {path.stem}: max {int(counts.max())} > {args.max_balance}x ideal {ideal:.1f}"}))
            return
        err = oracle_topk_error(activations=activations, down=down, assignment=assignment, top_k=args.top_k)
        per_layer.append({"layer": path.stem, "error": err})

    mean_error = float(np.mean([item["error"] for item in per_layer]))
    print(json.dumps({"mean_error": mean_error, "per_layer": per_layer}, indent=2))


if __name__ == "__main__":
    main()
