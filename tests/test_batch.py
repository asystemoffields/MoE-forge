from __future__ import annotations

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
    assert manifest["comparison"]["status"] == "written"
    assert manifest["recovery_eval"]["enabled"] is True
    assert compare["report_count"] == 2
    assert saved_manifest["sample_source"] == {"kind": "input_ids", "sample_count": 1}
    assert (output_dir / "eval-all.html").exists()
    assert (output_dir / "eval-compare.html").exists()


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
