from __future__ import annotations

import json
from pathlib import Path

import pytest

from moeforge.cli import main
from moeforge.conversion import ConversionRunOptions, run_conversion

torch = pytest.importorskip("torch")


def test_convert_cli_builds_native_wrapper_package(tmp_path: Path) -> None:
    transformers = pytest.importorskip("transformers")
    model_dir = _write_tiny_llama_checkpoint(tmp_path / "tiny-llama", transformers=transformers)
    run_dir = tmp_path / "run"

    status = main(
        [
            "convert",
            str(model_dir),
            "--output-dir",
            str(run_dir),
            "--moe-layers",
            "all",
            "--experts",
            "2",
            "--top-k",
            "1",
            "--shared-ratio",
            "0.25",
            "--token-router-top-k",
            "1",
            "--recover",
            "--recover-steps",
            "1",
            "--train-input-ids-json",
            "[[1,2,3,4]]",
            "--eval-input-ids-json",
            "[[1,2,3,4]]",
            "--eval-expert-mode",
            "all",
            "--eval-expert-mode",
            "learned-router",
        ]
    )

    report = json.loads((run_dir / "convert-report.json").read_text(encoding="utf-8"))
    loaded = transformers.AutoModelForCausalLM.from_pretrained(run_dir / "recovery-experiment" / "recovered-wrapper")

    assert status == 0
    assert report["format"] == "moeforge_conversion_run"
    assert report["status"] == "publish_ready"
    assert report["preflight"]["status"] == "ready"
    assert report["publish_readiness"]["status"] == "ready"
    assert report["artifacts"]["recovered_wrapper"] == str(run_dir / "recovery-experiment" / "recovered-wrapper")
    assert (run_dir / "publish-readiness.json").exists()
    assert (run_dir / "recovery-experiment" / "recovered-wrapper" / "config.json").exists()
    assert (run_dir / "wrapper" / "source-model" / "config.json").exists()
    assert (run_dir / "recovery-experiment" / "recovered-wrapper" / "MODEL_CARD.md").exists()
    assert loaded.config.model_type == "moeforge_carved_moe"


def test_convert_dry_run_reports_next_artifacts(tmp_path: Path) -> None:
    transformers = pytest.importorskip("transformers")
    model_dir = _write_tiny_llama_checkpoint(tmp_path / "tiny-llama", transformers=transformers)
    report = run_conversion(
        ConversionRunOptions(
            model=str(model_dir),
            output_dir=tmp_path / "dry",
            experts=2,
            top_k=1,
            shared_ratio=0.25,
            dry_run=True,
        )
    )

    assert report["status"] == "dry_run"
    assert report["passed"]
    assert (tmp_path / "dry" / "recipe.json").exists()
    assert (tmp_path / "dry" / "carve-manifest.json").exists()
    assert (tmp_path / "dry" / "carved" / "carve-apply-dry-run.json").exists()
    assert not (tmp_path / "dry" / "wrapper").exists()


def _write_tiny_llama_checkpoint(path: Path, *, transformers):
    torch.manual_seed(1234)
    config = transformers.LlamaConfig(
        attention_bias=False,
        hidden_size=8,
        intermediate_size=16,
        max_position_embeddings=16,
        num_attention_heads=2,
        num_hidden_layers=2,
        num_key_value_heads=2,
        tie_word_embeddings=False,
        vocab_size=32,
    )
    model = transformers.LlamaForCausalLM(config)
    model.save_pretrained(path, safe_serialization=True)
    return path
