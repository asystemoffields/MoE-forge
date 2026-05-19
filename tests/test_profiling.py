from __future__ import annotations

from pathlib import Path

from moeforge.adapters import ADAPTERS
from moeforge.profiling import (
    ChannelStats,
    ProfileOptions,
    load_calibration_texts,
    resolve_profile_modules,
)


def test_channel_stats_accumulates_nested_rows() -> None:
    stats = ChannelStats(threshold=0.5)

    stats.update([[[1.0, -2.0, 0.25], [0.0, 4.0, -0.75]]])
    report = stats.to_report(include_vectors=True, top_k_channels=2)

    assert report["count"] == 2
    assert report["width"] == 3
    assert report["vectors"]["mean_abs"] == [0.5, 3.0, 0.5]
    assert report["vectors"]["active_rate"] == [0.5, 1.0, 0.5]
    assert report["top_channels"][0]["channel"] == 1


def test_channel_stats_assigns_shared_and_experts() -> None:
    stats = ChannelStats()
    stats.update([[10.0, 8.0, 1.0, 1.0], [10.0, 4.0, 2.0, 1.0]])

    assignment = stats.assign_experts(experts=2, shared_ratio=0.25)

    assert assignment["available"] is True
    assert assignment["shared_channels"] == [0]
    routed = [set(item["channels"]) for item in assignment["experts"]]
    assert set.union(*routed) == {1, 2, 3}


def test_resolve_profile_modules_for_llama_layers() -> None:
    adapter = next(item for item in ADAPTERS if item.family == "llama")

    targets = resolve_profile_modules(
        adapter=adapter,
        layer_count=4,
        layers="1:2",
        roles=("gate", "up"),
    )

    assert "model.layers.1.mlp.gate_proj" in targets
    assert "model.layers.2.mlp.up_proj" in targets
    assert "model.layers.0.mlp.gate_proj" not in targets


def test_load_calibration_texts_splits_blank_lines(tmp_path: Path) -> None:
    path = tmp_path / "calib.txt"
    path.write_text("first\n\nsecond\n\nthird", encoding="utf-8")

    samples = load_calibration_texts(text_file=path, max_samples=2)

    assert samples == ["first", "second"]


def test_profile_options_defaults() -> None:
    options = ProfileOptions()

    assert options.roles == ("gate", "up")
    assert options.sequence_length == 512
    assert options.experts == 8
