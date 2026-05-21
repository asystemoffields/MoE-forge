"""Write a tiny synthetic router-ready layer .npz so eval_router.py can be exercised without
the dense model. Keys match the updated capture_layer.py: hidden, gate, up, down, activations,
importance. Deterministic. For tests/demos only -- real runs use captured SmolLM layers."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from moeforge.grouping import channel_importance, intermediate_activations


def make_fixture(hidden_dim=32, intermediate=64, tokens=80, seed=0):
    rng = np.random.default_rng(seed)
    # Structured hidden states (a few latent directions) so experts differentiate -> routing matters.
    latent = rng.standard_normal((tokens, 8))
    basis = rng.standard_normal((8, hidden_dim))
    hidden = (latent @ basis + 0.3 * rng.standard_normal((tokens, hidden_dim))).astype(np.float32)
    scale = 1.0 / np.sqrt(hidden_dim)
    gate = (rng.standard_normal((intermediate, hidden_dim)) * scale).astype(np.float32)
    up = (rng.standard_normal((intermediate, hidden_dim)) * scale).astype(np.float32)
    down = (rng.standard_normal((hidden_dim, intermediate)) * scale).astype(np.float32)
    activations = intermediate_activations(hidden.astype(np.float64), gate.astype(np.float64), up.astype(np.float64)).astype(np.float32)
    importance = channel_importance(activations).astype(np.float32)
    return dict(hidden=hidden, gate=gate, up=up, down=down, activations=activations, importance=importance)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("examples/router-search/fixture.npz"))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    fixture = make_fixture(seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **fixture)
    print(f"wrote {args.output}  tokens={fixture['hidden'].shape[0]} hidden={fixture['hidden'].shape[1]} "
          f"channels={fixture['activations'].shape[1]}")


if __name__ == "__main__":
    main()
