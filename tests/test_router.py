from __future__ import annotations

import json
from pathlib import Path

import pytest

from moeforge.carve import build_carve_manifest
from moeforge.materialize import materialize_carve_manifest
from moeforge.router import build_router_plan, select_expert_pool
from moeforge.runtime import CarvedGatedMLP

torch = pytest.importorskip("torch")
safetensors_torch = pytest.importorskip("safetensors.torch")


def test_build_router_plan_from_document_profile(tmp_path: Path) -> None:
    profile_path = _write_profile(tmp_path)

    plan = build_router_plan(profile_path=profile_path, pool_size=1)
    payload = plan.to_dict()

    assert payload["strategy"] == "document_pool_then_token_router"
    assert payload["expert_count"] == 3
    assert payload["pool_size"] == 1
    assert payload["documents"][0]["experts"] == [1]
    assert payload["documents"][1]["experts"] == [2]
    assert payload["default_pool"] == [1]
    assert "https://allenai.org/blog/emo" in payload["references"]


def test_select_expert_pool_uses_hash_index_then_default(tmp_path: Path) -> None:
    plan = build_router_plan(profile_path=_write_profile(tmp_path), pool_size=1).to_dict()

    assert select_expert_pool(plan, text_sha256="doc-b") == [2]
    assert select_expert_pool(plan, document_index=0) == [1]
    assert select_expert_pool(plan, text_sha256="unknown") == plan["default_pool"]


def test_runtime_forward_with_router_uses_selected_pool(tmp_path: Path) -> None:
    model = _write_checkpoint(tmp_path / "model")
    manifest_path = _write_manifest(tmp_path, model)
    output_dir = tmp_path / "out"
    materialize_carve_manifest(manifest_path=manifest_path, output_dir=output_dir)
    router_plan = {
        "default_pool": [0, 1],
        "documents": [
            {"document_index": 0, "text_sha256": "doc-a", "experts": [0]},
            {"document_index": 1, "text_sha256": "doc-b", "experts": [1]},
        ],
    }
    runtime = CarvedGatedMLP.from_artifact(
        manifest_path=manifest_path,
        artifact_path=output_dir / "carved-experts.safetensors",
        layer=0,
    )
    x = torch.tensor([[0.25, -0.5]], dtype=torch.float32)

    doc_a = runtime.forward_with_router(x, router_plan=router_plan, text_sha256="doc-a")
    doc_b = runtime.forward_with_router(x, router_plan=router_plan, text_sha256="doc-b")

    assert doc_a.shape == doc_b.shape
    assert not torch.allclose(doc_a, doc_b)


def _write_profile(tmp_path: Path) -> Path:
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(
        json.dumps(
            {
                "adapter_family": "llama",
                "modules": {
                    "m": {
                        "assignment": {
                            "experts": [
                                {"expert": 0, "channels": [0]},
                                {"expert": 1, "channels": [1]},
                                {"expert": 2, "channels": [2]},
                            ]
                        }
                    }
                },
                "documents": [
                    {
                        "index": 0,
                        "text_sha256": "doc-a",
                        "expert_pool": {
                            "method": "top_channel_mod_expert_score",
                            "experts": [1],
                            "scores": [0.0, 3.0, 1.0],
                        },
                    },
                    {
                        "index": 1,
                        "text_sha256": "doc-b",
                        "expert_pool": {
                            "method": "top_channel_mod_expert_score",
                            "experts": [2],
                            "scores": [1.0, 0.5, 2.0],
                        },
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    return profile_path


def _write_checkpoint(path: Path) -> Path:
    path.mkdir(parents=True)
    (path / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["LlamaForCausalLM"],
                "model_type": "llama",
                "hidden_size": 2,
                "intermediate_size": 4,
                "num_hidden_layers": 1,
            }
        ),
        encoding="utf-8",
    )
    safetensors_torch.save_file(
        {
            "model.layers.0.mlp.gate_proj.weight": torch.arange(8, dtype=torch.float32).reshape(4, 2),
            "model.layers.0.mlp.up_proj.weight": torch.arange(100, 108, dtype=torch.float32).reshape(4, 2),
            "model.layers.0.mlp.down_proj.weight": torch.arange(200, 208, dtype=torch.float32).reshape(2, 4),
        },
        str(path / "model.safetensors"),
    )
    return path


def _write_manifest(tmp_path: Path, model: Path) -> Path:
    recipe_path = tmp_path / "recipe.json"
    recipe_path.write_text(
        json.dumps(
            {
                "strategy": "carved_mlp",
                "experts": 2,
                "shared_ratio": 0.25,
                "moe_layers": [0],
                "layout": {
                    "layers": [
                        {
                            "layer": 0,
                            "intermediate_size": 4,
                            "shared_channels": 1,
                            "expert_channels": [2, 1],
                        }
                    ]
                },
            }
        ),
        encoding="utf-8",
    )
    manifest = build_carve_manifest(model=str(model), recipe_path=recipe_path)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest.to_dict()), encoding="utf-8")
    return manifest_path

