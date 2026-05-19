from __future__ import annotations

import json
from pathlib import Path

from moeforge.cli import main
from moeforge.model_card import write_model_card


def test_write_model_card_summarizes_wrapper_reports_and_commands(tmp_path: Path) -> None:
    wrapper = _write_wrapper(tmp_path / "wrapper")
    eval_report = tmp_path / "eval-report.json"
    recovery_report = tmp_path / "recovery-report.json"
    validation_report = tmp_path / "validation-report.json"
    _write_json(eval_report, _eval_report())
    _write_json(recovery_report, _recovery_report())
    _write_json(validation_report, _validation_report())

    output = tmp_path / "MODEL_CARD.md"
    summary = write_model_card(
        wrapper_dir=wrapper,
        output_path=output,
        eval_reports=[eval_report],
        recovery_reports=[recovery_report],
        validation_reports=[validation_report],
        commands=["moe-forge eval-hf model --wrapper wrapper --expert-mode learned-router"],
    )

    card = output.read_text(encoding="utf-8")
    assert summary["format"] == "moeforge_model_card"
    assert summary["eval_report_count"] == 1
    assert "# MoE Forge Model Card" in card
    assert "Token router top-k: `2`" in card
    assert "learned-router" in card
    assert "`eval-report.json`" in card
    assert "`recovery-report.json`" in card
    assert "`validation-report.json`" in card
    assert "Updated Router Tensors" in card
    assert "| `recovery-report.json` | 3 | 1.2 | 0.7 | 12 | 4 | validated |" in card
    assert "loaded / 2 layers / 2 routers" in card
    assert "4/4; missing 0" in card
    assert "### Router Activity" in card
    assert "L0:[0,1]" in card
    assert "0:4, 1:4" in card
    assert "0:0.6, 1:0.4" in card
    assert "eval warning" in card
    assert "moe-forge eval-hf model --wrapper wrapper --expert-mode learned-router" in card


def test_model_card_cli_writes_markdown(tmp_path: Path) -> None:
    wrapper = _write_wrapper(tmp_path / "wrapper")
    eval_report = tmp_path / "eval-report.json"
    output = tmp_path / "card.md"
    _write_json(eval_report, _eval_report())

    status = main(
        [
            "model-card",
            "--wrapper",
            str(wrapper),
            "--eval-report",
            str(eval_report),
            "--output",
            str(output),
            "--command",
            "moe-forge wrapper-export --token-router-top-k 2",
        ]
    )

    assert status == 0
    card = output.read_text(encoding="utf-8")
    assert "## Reproduction Commands" in card
    assert "wrapper-export --token-router-top-k 2" in card


def _write_wrapper(path: Path) -> Path:
    path.mkdir(parents=True)
    _write_json(
        path / "moeforge_config.json",
        {
            "format_version": 1,
            "model_type": "moeforge_carved_moe",
            "adapter_family": "llama",
            "source_model": "source-model",
            "manifest_path": "carve-manifest.json",
            "artifact_path": "carved-experts.safetensors",
            "router_plan_path": "router-plan.json",
            "token_router_top_k": 2,
            "token_router_path": "learned-router.safetensors",
            "activation": "silu",
            "expert_count": 3,
            "layers": [
                {
                    "layer": 0,
                    "width": 16,
                    "tensor_prefix": "moe.layers.0.mlp",
                    "expert_count": 3,
                    "shared_channels": 4,
                    "expert_channels": [4, 4, 4],
                },
                {
                    "layer": 1,
                    "width": 16,
                    "tensor_prefix": "moe.layers.1.mlp",
                    "expert_count": 3,
                    "shared_channels": 4,
                    "expert_channels": [4, 4, 4],
                },
            ],
            "warnings": ["wrapper warning"],
            "references": ["https://github.com/asystemoffields/MoE-forge"],
        },
    )
    return path


def _eval_report() -> dict:
    return {
        "format": "moeforge_hf_eval",
        "model": "source-model",
        "package_dir": "wrapper",
        "passed": False,
        "sample_count": 1,
        "max_abs_error": 0.2,
        "samples": [{"expert_mode": "learned-router"}],
        "summary": {
            "average_teacher_kl_loss": 0.03,
            "average_nll_loss_delta": 0.1,
            "average_carved_vs_dense_latency_ratio": 0.8,
        },
        "active_experts": [
            {
                "layer": 0,
                "experts": [0, 1],
                "mode": "learned-router",
                "token_count": 4,
                "top_k": 2,
                "expert_token_counts": {"0": 4, "1": 4},
                "mean_selected_weight_by_expert": {"0": 0.6, "1": 0.4},
            }
        ],
        "warnings": ["eval warning"],
    }


def _recovery_report() -> dict:
    return {
        "format": "moeforge_recovery_experiment",
        "summary": {
            "steps_completed": 3,
            "initial_loss": 1.2,
            "final_loss": 0.7,
            "recovered_updated_tensor_count": 12,
            "recovered_updated_router_tensor_count": 4,
            "recovered_wrapper_validation_status": "validated",
        },
    }


def _validation_report() -> dict:
    return {
        "format": "moeforge_recovered_wrapper_validation",
        "status": "validated",
        "errors": [],
        "reload": {"loaded_layer_count": 2},
        "native_load": {"status": "loaded", "replaced_layer_count": 2, "token_router_layer_count": 2},
        "router_tensor_validation": {"tensor_count": 4, "expected_tensor_count": 4, "missing_expected": []},
        "tensor_comparison": {"changed_tensor_count": 12},
    }


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
