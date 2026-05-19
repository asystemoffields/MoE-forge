from __future__ import annotations

from pathlib import Path

from moeforge.adapters import ADAPTERS
from moeforge.profiling import (
    ActivationProfile,
    ChannelStats,
    DocumentStats,
    ProfileOptions,
    load_calibration_texts,
    recommend_document_expert_pool,
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


def test_document_stats_report_preserves_sample_identity() -> None:
    document = DocumentStats.from_text(index=2, text="alpha beta")

    document.update("model.layers.0.mlp.gate_proj", [[1.0, -2.0, 0.5]], threshold=0.0)
    report = document.to_report(
        module_targets={"model.layers.0.mlp.gate_proj": {"layer": 0, "role": "gate"}},
        include_vectors=False,
        top_k_channels=2,
        experts=2,
        pool_size=1,
    )

    assert report["index"] == 2
    assert report["char_count"] == 10
    assert len(report["text_sha256"]) == 64
    module = report["modules"]["model.layers.0.mlp.gate_proj"]
    assert module["target"] == {"layer": 0, "role": "gate"}
    assert [item["channel"] for item in module["top_channels"]] == [1, 0]
    assert report["expert_pool"]["experts"] == [1]


def test_activation_profile_tracks_per_document_stats() -> None:
    profile = ActivationProfile(
        model="tiny",
        adapter_family="llama",
        samples=2,
        sequence_length=16,
        module_targets={"m": {"layer": 0, "role": "gate"}},
    )

    profile.begin_document(index=0, text="first")
    profile.update("m", [[1.0, 0.0]], threshold=0.0)
    profile.end_document()
    profile.begin_document(index=1, text="second")
    profile.update("m", [[0.0, 2.0]], threshold=0.0)
    profile.end_document()
    report = profile.to_report(
        include_vectors=False,
        top_k_channels=2,
        include_document_vectors=True,
        document_top_k_channels=1,
        experts=2,
        shared_ratio=0.25,
        document_pool_size=1,
    )

    assert report["document_count"] == 2
    assert report["modules"]["m"]["count"] == 2
    assert report["documents"][0]["modules"]["m"]["vectors"]["mean_abs"] == [1.0, 0.0]
    assert report["documents"][1]["modules"]["m"]["top_channels"][0]["channel"] == 1
    assert report["documents"][1]["expert_pool"]["experts"] == [1]


def test_recommend_document_expert_pool_scores_top_channels() -> None:
    pool = recommend_document_expert_pool(
        module_reports={
            "a": {
                "top_channels": [
                    {"channel": 5, "mean_abs": 4.0},
                    {"channel": 2, "mean_abs": 1.0},
                ]
            }
        },
        experts=4,
        pool_size=2,
    )

    assert pool["experts"] == [1, 2]


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
    assert options.document_top_k_channels == 8
