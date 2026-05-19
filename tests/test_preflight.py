from __future__ import annotations

import json
from pathlib import Path

from moeforge.cli import main
from moeforge.preflight import run_preflight


def test_preflight_reports_ready_wrapper_with_agent_next_commands(tmp_path: Path) -> None:
    wrapper = _write_wrapper(tmp_path / "wrapper")
    report = run_preflight(wrapper=wrapper, output_path=tmp_path / "preflight.json")

    saved = json.loads((tmp_path / "preflight.json").read_text(encoding="utf-8"))
    assert report["format"] == "moeforge_preflight"
    assert report["status"] == "ready"
    assert saved["artifacts"]["wrapper"]["layer_count"] == 2
    assert saved["warning_count"] >= 1
    assert any("eval-batch" in command for command in saved["next_commands"])
    assert any(check["name"] == "wrapper.router_training" for check in saved["checks"])


def test_preflight_blocks_invalid_recipe_and_cli_returns_nonzero(tmp_path: Path) -> None:
    recipe = tmp_path / "recipe.json"
    output = tmp_path / "preflight.json"
    recipe.write_text(json.dumps({"moe_layers": []}), encoding="utf-8")

    status = main(["preflight", "--recipe", str(recipe), "--output", str(output)])

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert status == 1
    assert payload["status"] == "blocked"
    assert payload["failed_check_count"] >= 2
    assert any(check["name"] == "recipe.experts" for check in payload["checks"])


def _write_wrapper(path: Path) -> Path:
    path.mkdir(parents=True)
    (path / "carved-experts.safetensors").write_bytes(b"placeholder")
    (path / "moeforge_config.json").write_text(
        json.dumps(
            {
                "format_version": 1,
                "model_type": "moeforge_carved_moe",
                "adapter_family": "llama",
                "source_model": "source-model",
                "manifest_path": "carve-manifest.json",
                "artifact_path": "carved-experts.safetensors",
                "router_plan_path": None,
                "token_router_top_k": 2,
                "token_router_path": None,
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
                "warnings": [],
                "references": [],
            }
        ),
        encoding="utf-8",
    )
    return path
