from __future__ import annotations

import numpy as np

from moeforge.grouping import (
    SHARED,
    balanced_assign,
    balanced_grouping,
    magnitude_grouping,
    oracle_topk_error,
    random_grouping,
)


def _separable_case():
    # 4 channels, identity down so output dim i carries channel i.
    # Tokens 0-1 fire channels {0,1}; tokens 2-3 fire channels {2,3}.
    activations = np.array(
        [[5.0, 5.0, 0.0, 0.0], [5.0, 5.0, 0.0, 0.0], [0.0, 0.0, 5.0, 5.0], [0.0, 0.0, 5.0, 5.0]]
    )
    down = np.eye(4)
    return activations, down


def test_full_topk_reconstructs_exactly() -> None:
    activations, down = _separable_case()
    assignment = np.array([0, 1, 2, 3])  # 4 experts, any grouping
    err = oracle_topk_error(activations=activations, down=down, assignment=assignment, top_k=4)
    assert err < 1e-9


def test_all_shared_reconstructs_exactly() -> None:
    activations, down = _separable_case()
    assignment = np.full(4, SHARED)
    err = oracle_topk_error(activations=activations, down=down, assignment=assignment, top_k=1)
    assert err < 1e-9


def test_grouping_quality_is_measurable() -> None:
    activations, down = _separable_case()
    good = np.array([0, 0, 1, 1])  # co-firing channels share an expert
    bad = np.array([0, 1, 0, 1])  # co-firing channels split across experts
    good_err = oracle_topk_error(activations=activations, down=down, assignment=good, top_k=1)
    bad_err = oracle_topk_error(activations=activations, down=down, assignment=bad, top_k=1)
    assert good_err < 1e-9
    assert bad_err > good_err
    assert bad_err > 0.1


def test_magnitude_grouping_shapes_and_shared_count() -> None:
    importance = np.arange(16, dtype=float)
    assignment = magnitude_grouping(importance, n_experts=4, shared_ratio=0.25)
    assert assignment.shape == (16,)
    assert int((assignment == SHARED).sum()) == 4
    # The top-4 importance channels (12..15) are the shared ones.
    assert set(np.where(assignment == SHARED)[0]) == {12, 13, 14, 15}
    experts = assignment[assignment != SHARED]
    assert set(experts.tolist()) == {0, 1, 2, 3}


def test_random_grouping_is_valid() -> None:
    rng = np.random.default_rng(0)
    assignment = random_grouping(20, n_experts=4, shared_ratio=0.2, rng=rng)
    assert assignment.shape == (20,)
    assert int((assignment == SHARED).sum()) == 4
    assert set(assignment[assignment != SHARED].tolist()).issubset({0, 1, 2, 3})


def test_balanced_assign_produces_equal_sizes() -> None:
    rng = np.random.default_rng(0)
    points = rng.normal(size=(64, 5))
    labels = balanced_assign(points, 4, rng=rng)
    sizes = [int((labels == c).sum()) for c in range(4)]
    assert sum(sizes) == 64
    assert max(sizes) - min(sizes) <= 1  # near-equal by construction


def test_balanced_grouping_is_balanced_and_valid() -> None:
    rng = np.random.default_rng(0)
    activations = rng.normal(size=(40, 24))
    importance = np.abs(activations).mean(axis=0)
    assignment = balanced_grouping(activations, importance, n_experts=4, shared_ratio=0.25, rng=rng, transform="abs")
    assert assignment.shape == (24,)
    assert int((assignment == SHARED).sum()) == 6
    sizes = [int((assignment == c).sum()) for c in range(4)]
    ideal = sum(sizes) / 4
    assert max(sizes) <= 1.5 * ideal  # genuinely balanced, well under the 2x cap
