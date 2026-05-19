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
        validator=_fake_validator,
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
    assert report["summary"]["recovered_wrapper_validation_status"] == "validated"
    assert report["summary"]["recovered_updated_tensor_count"] == 1
    assert report["summary"]["improved_modes_by_teacher_kl"] == 2
    assert report["summary"]["total_loss_delta"] == -0.75
    assert report["quality_trends"]["training"]["final_teacher_kl"] == 0.05
    assert report["quality_trends"]["before_after_quality"]["average_teacher_kl_delta"] < 0
    assert report["quality_trends"]["before_after_quality"]["best_nll_delta_mode"]["expert_mode"] in {"all", "router"}
    assert comparison["mode_deltas"][0]["max_abs_error_delta"] < 0
    assert comparison["mode_deltas"][0]["teacher_kl_loss_delta"] < 0
    assert Path(report["artifacts"]["html_report"]).exists()
    assert Path(report["artifacts"]["recovered_wrapper_validation"]).exists()
    assert Path(report["artifacts"]["before_eval_manifest"]).exists()
    assert Path(report["artifacts"]["after_eval_manifest"]).exists()


def test_run_recovery_experiment_resolves_relative_config_paths(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    Path("experiment.json").write_text(
        json.dumps(
            {
                "model": "tiny-model",
                "wrapper": "wrapper",
                "output_dir": "experiment",
                "eval": {"expert_modes": ["all"], "input_ids": [[1, 2, 3]]},
                "train": {"input_ids": [[1, 2, 3]]},
                "recovery": {"schedule": {"steps": 1}},
            }
        ),
        encoding="utf-8",
    )

    run_recovery_experiment(
        config_path=Path("experiment.json"),
        evaluator=_fake_evaluator,
        recovery_runner=_fake_recovery_runner,
        exporter=_fake_exporter,
        validator=_fake_validator,
    )

    before_config = json.loads(Path("experiment/before/eval-batch-config.json").read_text(encoding="utf-8"))
    recovery_config = json.loads(Path("experiment/recovery/recovery-config.json").read_text(encoding="utf-8"))
    assert before_config["model"] == str(tmp_path / "tiny-model")
    assert before_config["wrapper"] == str(tmp_path / "wrapper")
    assert recovery_config["teacher_model"] == str(tmp_path / "tiny-model")
    assert recovery_config["wrapper"] == str(tmp_path / "wrapper")


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
                "average_teacher_kl_loss": self.max_abs / 10,
                "average_dense_nll_loss": 2.0,
                "average_carved_nll_loss": 2.0 + self.max_abs,
                "average_nll_loss_delta": self.max_abs,
                "loss_token_count": 2,
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
                    "teacher_kl_loss": self.max_abs / 10,
                    "dense_nll_loss": 2.0,
                    "carved_nll_loss": 2.0 + self.max_abs,
                    "nll_loss_delta": self.max_abs,
                    "loss_token_count": 2,
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
        "losses": [
            {
                "step": 1,
                "total_loss": 1.0,
                "teacher_kl": 0.2,
                "logits_mse": 0.1,
                "z_loss": 0.0,
                "learning_rate": 0.001,
            },
            {
                "step": max_steps or 2,
                "total_loss": 0.25,
                "teacher_kl": 0.05,
                "logits_mse": 0.025,
                "z_loss": 0.0,
                "learning_rate": 0.001,
            },
        ],
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


def _fake_validator(
    *,
    source_wrapper: Path,
    recovered_wrapper: Path,
    checkpoint_path: Path,
    export_report_path: Path,
    output_path: Path,
) -> dict:
    report = {
        "format": "moeforge_recovered_wrapper_validation",
        "source_wrapper": str(source_wrapper),
        "recovered_wrapper": str(recovered_wrapper),
        "checkpoint_path": str(checkpoint_path),
        "export_report_path": str(export_report_path),
        "status": "validated",
        "passed": True,
        "config_checks": {"layer_signature_match": True},
        "tensor_comparison": {
            "source_tensor_count": 2,
            "recovered_tensor_count": 2,
            "updated_tensor_count": 1,
            "changed_tensor_count": 1,
            "missing_from_recovered": [],
            "extra_in_recovered": [],
            "updated_tensors": [
                {
                    "tensor": "moe.layers.0.mlp.experts.0.gate.weight",
                    "shape": [1, 2],
                    "source_dtype": "float32",
                    "recovered_dtype": "float32",
                    "max_abs_delta": 0.1,
                    "mean_abs_delta": 0.05,
                }
            ],
        },
        "reload": {"loaded_layer_count": 1, "loaded_layers": [{"layer": 0}]},
        "errors": [],
        "warnings": [],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report), encoding="utf-8")
    return report
