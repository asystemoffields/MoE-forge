"""Map the quality-vs-sparsity frontier for carved MoE channel groupings.

For captured FFN layers, sweep top_k (and shared_ratio) and report oracle-top-k reconstruction
error against the *active fraction* (the share of FFN channels active per token = the compute
cost). This is the UN-recovered, perfect-router frontier — the floor before recovery training
bends it. It shows the raw quality/compute tradeoff and whether a sparsity "knee" exists.

Usage:
  python examples/grouping-search/sparsity_frontier.py \
      --layers examples/grouping-search/layer3.npz examples/grouping-search/layer6.npz examples/grouping-search/layer9.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from moeforge.grouping import balanced_grouping, oracle_topk_error


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layers", type=Path, nargs="+", required=True)
    parser.add_argument("--experts", type=int, default=8)
    parser.add_argument("--shared-ratio", type=float, default=0.125)
    parser.add_argument("--transform", default="abs", choices=["raw", "abs", "squared"])
    args = parser.parse_args()

    layers = [np.load(p) for p in args.layers]

    def mean_error(n_experts: int, shared_ratio: float, top_k: int) -> float:
        errors = []
        for data in layers:
            activations = data["activations"].astype(np.float64)
            assignment = balanced_grouping(
                activations, data["importance"].astype(np.float64),
                n_experts=n_experts, shared_ratio=shared_ratio,
                rng=np.random.default_rng(0), transform=args.transform,
            )
            errors.append(oracle_topk_error(
                activations=activations, down=data["down"].astype(np.float64),
                assignment=assignment, top_k=top_k,
            ))
        return float(np.mean(errors))

    print(f"experts={args.experts} shared_ratio={args.shared_ratio} transform={args.transform}")
    print(f"\n{'top_k':>6}{'active_frac':>12}{'recon_err':>11}")
    for top_k in range(1, args.experts + 1):
        active = args.shared_ratio + (top_k / args.experts) * (1 - args.shared_ratio)
        print(f"{top_k:>6}{active:>12.3f}{mean_error(args.experts, args.shared_ratio, top_k):>11.4f}")

    print(f"\ntop_k=2 of {args.experts}: sweep shared_ratio")
    print(f"{'shared':>7}{'active_frac':>12}{'recon_err':>11}")
    for shared in (0.0, 0.125, 0.25, 0.375, 0.5):
        active = shared + (2 / args.experts) * (1 - shared)
        print(f"{shared:>7.3f}{active:>12.3f}{mean_error(args.experts, shared, 2):>11.4f}")


if __name__ == "__main__":
    main()
