from __future__ import annotations

import json
from pathlib import Path
import shutil

import modal


APP_NAME = "moeforge-smollm-recovery"
VOLUME_NAME = "moeforge-benchmarks"
MOEFORGE_REVISION = "21900998e27095a0a65517f32f57fbd1b61ed757"
REMOTE_ROOT = Path("/vol")


app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "torch>=2.2",
        "transformers>=4.45",
        "safetensors>=0.4",
        "accelerate>=0.34",
        "datasets>=3.5,<4",
        "sentencepiece",
        "protobuf",
        "pillow",
    )
    .pip_install(f"git+https://github.com/asystemoffields/MoE-forge.git@{MOEFORGE_REVISION}")
)


@app.function(image=image, gpu="T4", timeout=60 * 60 * 12, volumes={str(REMOTE_ROOT): volume})
def run_smollm_recovery(
    *,
    run_name: str,
    wrapper: str,
    source_model: str,
    steps: int,
    batch_size: int,
    sequence_length: int,
    learning_rate: float,
    train_experts: bool,
    corpus_sources: str,
    max_samples_per_source: int,
    include_answers: bool,
    token_router_top_k: int | None,
    router_oracle_method: str,
) -> dict[str, object]:
    from moeforge.corpus import CorpusBuildOptions, build_recovery_corpus
    from moeforge.recovery_experiment import run_recovery_experiment

    run_dir = REMOTE_ROOT / "recovery-runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    effective_wrapper = _prepared_wrapper(
        wrapper=Path(wrapper),
        source_model=source_model,
        run_dir=run_dir,
        token_router_top_k=token_router_top_k,
    )
    train_file = run_dir / "train.txt"
    corpus_manifest_path = run_dir / "recovery-corpus.json"
    corpus_manifest = build_recovery_corpus(
        CorpusBuildOptions(
            output_path=train_file,
            manifest_path=corpus_manifest_path,
            sources=tuple(item.strip() for item in corpus_sources.split(",") if item.strip()),
            max_samples_per_source=max_samples_per_source,
            seed=13,
            include_answers=include_answers,
            split="train",
        )
    )
    config = {
        "model": source_model,
        "wrapper": str(effective_wrapper),
        "output_dir": str(run_dir),
        "corpus": corpus_manifest,
        "train": {
            "text_file": str(train_file),
            "sequence_length": sequence_length,
        },
        "eval": {
            "text_file": str(train_file),
            "sequence_length": sequence_length,
            "expert_modes": ["all", "learned-router"],
            "device": "cuda",
            "write_html": True,
        },
        "recovery": {
            "trainable": {
                "experts": train_experts,
                "router": True,
                "shared": False,
                "dense_backbone": False,
            },
            "loss": {
                "teacher_kl_weight": 1.0,
                "logits_mse_weight": 0.05,
                "router_oracle_weight": 0.25,
                "router_oracle_method": router_oracle_method,
                "router_balance_weight": 0.01,
            },
            "optimizer": {
                "learning_rate": learning_rate,
                "weight_decay": 0.0,
            },
            "schedule": {
                "steps": steps,
                "batch_size": batch_size,
                "save_every_steps": steps,
                "eval_every_steps": max(1, steps // 4),
            },
        },
        "strict_validation": True,
    }
    config_path = run_dir / "recovery-experiment-config.json"
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report = run_recovery_experiment(config_path=config_path, output_dir=run_dir)
    (run_dir / "modal-recovery-manifest.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    volume.commit()
    return {
        "format": "moeforge_modal_recovery_manifest",
        "run_name": run_name,
        "run_dir": str(run_dir),
        "report": str(run_dir / "modal-recovery-manifest.json"),
        "recovered_wrapper": report.get("recovered_wrapper"),
        "wrapper": str(effective_wrapper),
        "token_router_top_k": token_router_top_k,
        "router_oracle_method": router_oracle_method,
        "summary": report.get("summary"),
        "artifacts": report.get("artifacts"),
        "corpus_manifest": str(corpus_manifest_path),
        "corpus_summary": {
            "sample_count": corpus_manifest.get("sample_count"),
            "sha256": corpus_manifest.get("sha256"),
            "sources": [
                {
                    "name": source.get("name"),
                    "status": source.get("status"),
                    "sample_count": source.get("sample_count"),
                }
                for source in corpus_manifest.get("sources", [])
                if isinstance(source, dict)
            ],
            "warnings": corpus_manifest.get("warnings"),
        },
    }


def _prepared_wrapper(
    *,
    wrapper: Path,
    source_model: str,
    run_dir: Path,
    token_router_top_k: int | None,
) -> Path:
    if token_router_top_k is None:
        return wrapper
    if token_router_top_k <= 0:
        raise ValueError("token_router_top_k must be positive")
    target = run_dir / f"source-wrapper-topk-{token_router_top_k}"
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)
    skip_dirs = {"source-model", "__pycache__"}
    for item in wrapper.iterdir():
        if item.name in skip_dirs or item.name.startswith("results_"):
            continue
        if item.is_dir():
            if item.name.startswith("20") and "T" in item.name:
                continue
            shutil.copytree(item, target / item.name)
        else:
            shutil.copy2(item, target / item.name)
    for config_name in ("config.json", "moeforge_config.json"):
        path = target / config_name
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["token_router_top_k"] = int(token_router_top_k)
        payload["source_model"] = source_model
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target


@app.local_entrypoint()
def main(
    run_name: str = "smollm-benchmix-router-1000",
    wrapper: str = "/vol/smollm-moe-v5",
    source_model: str = "/vol/smollm-moe-v5/source-model",
    steps: int = 1000,
    batch_size: int = 2,
    sequence_length: int = 192,
    learning_rate: float = 5e-5,
    train_experts: bool = False,
    corpus_sources: str = "smollm-benchmix,builtin-smoke",
    max_samples_per_source: int = 128,
    include_answers: bool = False,
    token_router_top_k: int | None = None,
    router_oracle_method: str = "magnitude",
) -> None:
    manifest = run_smollm_recovery.remote(
        run_name=run_name,
        wrapper=wrapper,
        source_model=source_model,
        steps=steps,
        batch_size=batch_size,
        sequence_length=sequence_length,
        learning_rate=learning_rate,
        train_experts=train_experts,
        corpus_sources=corpus_sources,
        max_samples_per_source=max_samples_per_source,
        include_answers=include_answers,
        token_router_top_k=token_router_top_k,
        router_oracle_method=router_oracle_method,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
