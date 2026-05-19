from __future__ import annotations

import json
from pathlib import Path

from moeforge.recovery_experiment import run_recovery_experiment


def test_run_recovery_experiment_orchestrates_before_recover_after(tmp_path: Path) -> None:
    config_path = tmp_path / "experiment.json"
    output_dir = tmp_path / "experiment"
    config_path.write_text(
        json.dumps(
            {
                "model": str(tmp_path / "tiny-model"),
                "wrapper": str(tmp_path / "wrapper"),
                "output_dir": str(output_dir),
                "eval": {
                    "expert_modes": ["all", "router"],
                    "input_ids": [[1, 2, 3]],
                    "write_html": True,
                },
                "train": {"input_ids": [[1, 2, 3]]},
                "recovery": {
                    "schedule": {"steps": 2},
                    "loss": {"teacher_kl_weight": 1.0},
                },
            }
        ),
        encoding="utf-8",
    )

    report = run_recovery_experiment(
        config_path=config_path,
        evaluator=_fake_evaluator,
        recovery_runner=_fake_recovery_runner,
        exporter=_fake_exporter,
    )

    saved = json.loads((output_dir / "recovery-experiment-report.json").read_text(encoding="utf-8"))
    before_config = json.loads((output_dir / "before" / "eval-batch-config.json").read_text(encoding="utf-8"))
    after_config = json.loads((output_dir / "after" / "eval-batch-config.json").read_text(encoding="utf-8"))
    comparison = json.loads((output_dir / "recovery-before-after.json").read_text(encoding="utf-8"))
    assert report["format"] == "moeforge_recovery_experiment"
    assert saved["summary"]["improved_modes_by_max_abs_error"] == 2
    assert before_config["wrapper"] == str(tmp_path / "wrapper")
    assert after_config["wrapper"] == str(output_dir / "recovered-wrapper")
    assert report["summary"]["initial_loss"] == 1.0
    assert report["summary"]["final_loss"] == 0.25
    assert comparison["mode_deltas"][0]["max_abs_error_delta"] < 0
    assert Path(report["artifacts"]["html_report"]).exists()
    assert Path(report["artifacts"]["before_eval_manifest"]).exists()
    assert Path(report["artifacts"]["after_eval_manifest"]).exists()


def _fake_evaluator(**kwargs):
    mode = kwargs["expert_mode"]
    recovered = "recovered-wrapper" in str(kwargs["package_dir"])
    base_error = 0.2 if mode == "all" else 0.4
    return _FakeEvalReport(mode=mode, max_abs=base_error / 2 if recovered else base_error)


class _FakeEvalReport:
    def __init__(self, *, mode: str, max_abs: float) -> None:
        self.mode = mode
        self.max_abs = max_abs

    def to_dict(self) -> dict:
        return {
            "model": "tiny-model",
            "package_dir": "wrapper",
            "source_model": "dense-source",
            "adapter_family": "llama",
            "sample_count": 1,
            "passed": self.max_abs == 0.0,
            "max_abs_error": self.max_abs,
            "mean_abs_error": self.max_abs / 2,
            "warnings": [],
            "summary": {
                "average_dense_latency_s": 0.01,
                "average_carved_latency_s": 0.02,
                "average_carved_vs_dense_latency_ratio": 2.0,
                "worst_layer": 0,
                "worst_layer_selected_vs_all_max_abs_error": self.max_abs,
            },
            "samples": [
                {
                    "index": 0,
                    "source": "input_ids:0",
                    "expert_mode": self.mode,
                    "max_abs_error": self.max_abs,
                    "mean_abs_error": self.max_abs / 2,
                    "carved_vs_dense_latency_ratio": 2.0,
                    "allclose": False,
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


def _fake_recovery_runner(*, plan_path: Path, output_path: Path, max_steps: int | None = None) -> dict:
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    output_dir = Path(plan["output_dir"])
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "format": "moeforge_recovery_checkpoint",
        "step": max_steps or 2,
        "metadata_path": str(checkpoint_dir / "checkpoint-step-2.json"),
        "state_path": str(checkpoint_dir / "trainable-state-step-2.pt"),
        "promoted_carved_parameters": [],
    }
    Path(checkpoint["metadata_path"]).write_text(json.dumps(checkpoint), encoding="utf-8")
    report = {
        "format": "moeforge_recovery_run",
        "output_dir": str(output_dir),
        "initial_loss": 1.0,
        "final_loss": 0.25,
        "steps_completed": max_steps or 2,
        "checkpoints": [checkpoint],
        "warnings": [],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report), encoding="utf-8")
    return report


def _fake_exporter(*, checkpoint_path: Path, wrapper_dir: Path, output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "format": "moeforge_recovery_export",
        "checkpoint_path": str(checkpoint_path),
        "source_wrapper": str(wrapper_dir),
        "output_dir": str(output_dir),
        "artifact_path": str(output_dir / "recovered-carved-experts.safetensors"),
        "updated_tensor_count": 1,
    }
    (output_dir / "recovery-export-report.json").write_text(json.dumps(report), encoding="utf-8")
    return report
