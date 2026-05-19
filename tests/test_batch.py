from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from moeforge.batch import EvalBatchError, run_eval_batch


def test_run_eval_batch_writes_reports_comparison_and_manifest(tmp_path: Path) -> None:
    config_path = tmp_path / "batch.json"
    config_path.write_text(
        json.dumps(
            {
                "model": "tiny-model",
                "wrapper": "wrapper",
                "output_dir": "batch-out",
                "expert_modes": ["all", "router"],
                "input_ids": [[1, 2, 3]],
                "write_html": True,
                "recovery_eval": {"enabled": True, "metrics": ["logits_parity"]},
            }
        ),
        encoding="utf-8",
    )

    manifest = run_eval_batch(config_path=config_path, evaluator=_fake_evaluator)

    output_dir = tmp_path / "batch-out"
    compare = json.loads((output_dir / "eval-compare.json").read_text(encoding="utf-8"))
    saved_manifest = json.loads((output_dir / "eval-batch-manifest.json").read_text(encoding="utf-8"))
    assert manifest["format"] == "moeforge_eval_batch"
    assert manifest["completed_report_count"] == 2
    assert manifest["runs"][0]["status"] == "passed"
    assert manifest["runs"][1]["status"] == "failed"
    assert manifest["runs"][1]["teacher_kl_loss"] == 0.05
    assert manifest["comparison"]["status"] == "written"
    assert manifest["recovery_eval"]["enabled"] is True
    assert compare["report_count"] == 2
    assert saved_manifest["sample_source"]["kind"] == "input_ids"
    assert saved_manifest["sample_source"]["sample_count"] == 1
    assert len(saved_manifest["sample_source"]["sha256"]) == 64
    assert len(saved_manifest["sample_source"]["sample_sha256"]) == 1
    assert (output_dir / "eval-all.html").exists()
    assert (output_dir / "eval-compare.html").exists()


def test_run_eval_batch_records_input_ids_file_identity(tmp_path: Path) -> None:
    dataset = tmp_path / "tokens.json"
    dataset.write_text(json.dumps([[7, 8, 9], [9, 8, 7]]), encoding="utf-8")
    config_path = tmp_path / "batch.json"
    config_path.write_text(
        json.dumps(
            {
                "model": "tiny-model",
                "wrapper": "wrapper",
                "output_dir": "batch-out",
                "expert_modes": ["all"],
                "input_ids_file": "tokens.json",
                "write_html": False,
            }
        ),
        encoding="utf-8",
    )
    seen: dict[str, object] = {}

    def evaluator(**kwargs):
        seen["input_ids"] = kwargs["input_ids"]
        return _FakeReport(mode=kwargs["expert_mode"], passed=True)

    manifest = run_eval_batch(config_path=config_path, evaluator=evaluator)

    source = manifest["sample_source"]
    assert seen["input_ids"] == [[7, 8, 9], [9, 8, 7]]
    assert source["kind"] == "input_ids"
    assert source["sample_count"] == 2
    assert source["input_ids_file"]["path"] == "tokens.json"
    assert source["input_ids_file"]["byte_count"] == len(dataset.read_bytes())
    assert source["input_ids_file"]["sha256"] == hashlib.sha256(dataset.read_bytes()).hexdigest()


def test_run_eval_batch_records_text_file_identity(tmp_path: Path) -> None:
    text_file = tmp_path / "samples.txt"
    text_file.write_text("alpha sample\n\nbeta sample", encoding="utf-8")
    config_path = tmp_path / "batch.json"
    config_path.write_text(
        json.dumps(
            {
                "model": "tiny-model",
                "wrapper": "wrapper",
                "output_dir": "batch-out",
                "expert_modes": ["all"],
                "text_file": "samples.txt",
                "write_html": False,
            }
        ),
        encoding="utf-8",
    )
    seen: dict[str, object] = {}

    def evaluator(**kwargs):
        seen["texts"] = kwargs["texts"]
        return _FakeReport(mode=kwargs["expert_mode"], passed=True)

    manifest = run_eval_batch(config_path=config_path, evaluator=evaluator)

    source = manifest["sample_source"]
    assert seen["texts"] == ["alpha sample", "beta sample"]
    assert source["kind"] == "text"
    assert source["sample_count"] == 2
    assert source["chunk_count"] == 2
    assert source["text_file"]["sha256"] == hashlib.sha256(text_file.read_bytes()).hexdigest()


def test_run_eval_batch_validates_modes(tmp_path: Path) -> None:
    config_path = tmp_path / "batch.json"
    config_path.write_text(
        json.dumps(
            {
                "model": "tiny-model",
                "wrapper": "wrapper",
                "expert_modes": ["sideways"],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(EvalBatchError, match="unsupported modes"):
        run_eval_batch(config_path=config_path, evaluator=_fake_evaluator)


def _fake_evaluator(**kwargs):
    mode = kwargs["expert_mode"]
    return _FakeReport(mode=mode, passed=mode == "all")


class _FakeReport:
    def __init__(self, *, mode: str, passed: bool) -> None:
        self.mode = mode
        self.passed = passed

    def to_dict(self) -> dict:
        max_abs = 0.0 if self.passed else 0.2
        return {
            "model": "tiny-model",
            "package_dir": "wrapper",
            "source_model": "dense-source",
            "adapter_family": "llama",
            "sample_count": 1,
            "passed": self.passed,
            "max_abs_error": max_abs,
            "mean_abs_error": max_abs / 2,
            "warnings": [],
            "summary": {
                "average_dense_latency_s": 0.01,
                "average_carved_latency_s": 0.01 if self.passed else 0.005,
                "average_carved_vs_dense_latency_ratio": 1.0 if self.passed else 0.5,
                "average_teacher_kl_loss": 0.0 if self.passed else 0.05,
                "average_dense_nll_loss": 2.0,
                "average_carved_nll_loss": 2.0 if self.passed else 2.2,
                "average_nll_loss_delta": 0.0 if self.passed else 0.2,
                "loss_token_count": 2,
                "worst_layer": 0,
                "worst_layer_selected_vs_all_max_abs_error": max_abs,
            },
            "samples": [
                {
                    "index": 0,
                    "source": "input_ids:0",
                    "expert_mode": self.mode,
                    "max_abs_error": max_abs,
                    "mean_abs_error": max_abs / 2,
                    "teacher_kl_loss": 0.0 if self.passed else 0.05,
                    "dense_nll_loss": 2.0,
                    "carved_nll_loss": 2.0 if self.passed else 2.2,
                    "nll_loss_delta": 0.0 if self.passed else 0.2,
                    "loss_token_count": 2,
                    "carved_vs_dense_latency_ratio": 1.0 if self.passed else 0.5,
                    "allclose": self.passed,
                }
            ],
            "active_experts": [
                {"sample_index": 0, "layer": 0, "mode": self.mode, "experts": [0]},
            ],
            "layer_attribution": [],
            "memory": {},
            "package": {"expert_count": 1},
            "replacements": {"replaced": []},
        }
