"""Tests for the router-search evaluator: metric parity with the library oracle, the seed
router landing between the oracle floor and random ceiling, and the state-budget guard."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "src"
sys.path.insert(0, str(SRC))

from moeforge.grouping import SHARED, oracle_topk_error  # noqa: E402

import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location("eval_router", Path(__file__).resolve().parent / "eval_router.py")
eval_router = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(eval_router)

from make_fixture import make_fixture  # noqa: E402  (same dir)


def _run(candidate: Path, layers: list[Path], **flags) -> dict:
    cmd = [sys.executable, str(Path(__file__).resolve().parent / "eval_router.py"),
           "--candidate", str(candidate)]
    cmd += ["--layers", *[str(p) for p in layers]]
    for key, value in flags.items():
        cmd += [f"--{key.replace('_', '-')}", str(value)]
    env = {"PYTHONPATH": str(SRC), "PYTHONIOENCODING": "utf-8"}
    import os
    completed = subprocess.run(cmd, env={**os.environ, **env}, text=True,
                               capture_output=True, encoding="utf-8")
    # The evaluator prints exactly one JSON object.
    return json.loads(completed.stdout)


def test_selection_error_matches_library_oracle():
    """selection_error with true contribution norms must reproduce oracle_topk_error exactly."""
    rng = np.random.default_rng(1)
    tokens, channels, hidden = 50, 40, 16
    activations = rng.standard_normal((tokens, channels))
    down = rng.standard_normal((hidden, channels))
    assignment = np.full(channels, SHARED)
    assignment[8:] = rng.integers(0, 4, size=channels - 8)  # 4 experts + shared
    top_k = 2

    dense = activations @ down.T
    shared_mask = assignment == SHARED
    base = activations[:, shared_mask] @ down[:, shared_mask].T
    contributions, norms = [], np.zeros((tokens, 4))
    for expert in range(4):
        mask = assignment == expert
        contribution = activations[:, mask] @ down[:, mask].T if mask.any() else np.zeros_like(dense)
        contributions.append(contribution)
        norms[:, expert] = np.linalg.norm(contribution, axis=1)

    ours = eval_router.selection_error(dense, base, contributions, norms, top_k)
    theirs = oracle_topk_error(activations=activations, down=down, assignment=assignment, top_k=top_k)
    assert ours == pytest.approx(theirs, rel=1e-9)


def test_seed_between_oracle_and_random(tmp_path):
    layer = tmp_path / "fixture.npz"
    np.savez_compressed(layer, **make_fixture(seed=0))
    seed = Path(__file__).resolve().parent / "candidates" / "seed_router.py"
    result = _run(seed, [layer], experts=4, top_k=2, shared_ratio=0.125)
    assert "error" not in result, result
    # Oracle is the floor, random the ceiling; a sane router lands in between (with slack).
    assert result["oracle"] <= result["mean_error"] + 1e-6
    assert result["mean_error"] <= result["random"] + 1e-6
    assert np.isfinite(result["mean_error"])
    assert result["per_layer"][0]["state_floats"] <= result["per_layer"][0]["state_budget"]


def test_state_budget_guard(tmp_path):
    """A candidate that smuggles a big matrix into state (to recompute activations) is rejected."""
    layer = tmp_path / "fixture.npz"
    np.savez_compressed(layer, **make_fixture(seed=0))
    cheater = tmp_path / "cheater.py"
    cheater.write_text(
        "import numpy as np\n"
        "def build_router(ctx, n_experts, top_k, rng):\n"
        "    return {'gate_copy': np.asarray(ctx['gate'])}\n"  # full [I,H] gate -> over budget
        "def route(hidden, state, n_experts, top_k):\n"
        "    return np.zeros((hidden.shape[0], n_experts))\n",
        encoding="utf-8",
    )
    result = _run(cheater, [layer], experts=4, top_k=2, state_budget_mult=8)
    assert "error" in result and "budget" in result["error"], result


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
