from __future__ import annotations

import json
import os
from pathlib import Path
import shutil

import modal


APP_NAME = "moeforge-smollm-recovery"
VOLUME_NAME = "moeforge-benchmarks"
HF_CACHE_VOLUME_NAME = "moeforge-hf-cache"
MOEFORGE_REVISION = "974eeb06a32290a39d87e4b97c3493b9c9d2ee1a"
REMOTE_ROOT = Path("/vol")
REMOTE_ROOT_DISPLAY = "/vol"
HF_CACHE_ROOT = "/cache"
# GPU is fixed at function-registration time; override per launch with MOEFORGE_GPU=H100/T4/L4.
DEFAULT_GPU = os.environ.get("MOEFORGE_GPU", "A10G")


app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
# Persistent HF datasets/hub cache so corpus builds stop re-downloading every run.
hf_cache_volume = modal.Volume.from_name(HF_CACHE_VOLUME_NAME, create_if_missing=True)

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
    .env(
        {
            "HF_HOME": f"{HF_CACHE_ROOT}/hf",
            "HF_DATASETS_CACHE": f"{HF_CACHE_ROOT}/hf/datasets",
        }
    )
)


@app.function(
    image=image,
    gpu=DEFAULT_GPU,
    timeout=60 * 60 * 12,
    volumes={str(REMOTE_ROOT): volume, HF_CACHE_ROOT: hf_cache_volume},
)
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
    eval_max_samples: int,
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
    # Decouple eval size from the (now potentially large) training corpus: eval on a fixed
    # small subset so the before/after eval stays fast as we scale training tokens/steps.
    eval_file = run_dir / "eval.txt"
    samples = [s for s in train_file.read_text(encoding="utf-8").split("\n\n") if s.strip()]
    eval_file.write_text("\n\n".join(samples[: max(1, eval_max_samples)]) + "\n", encoding="utf-8")
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
            "text_file": str(eval_file),
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
    hf_cache_volume.commit()
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
    batch_size: int = 8,
    sequence_length: int = 192,
    learning_rate: float = 5e-5,
    train_experts: bool = False,
    corpus_sources: str = "smollm-benchmix,builtin-smoke",
    max_samples_per_source: int = 128,
    include_answers: bool = False,
    token_router_top_k: int | None = None,
    router_oracle_method: str = "magnitude",
    eval_max_samples: int = 256,
    resume_from_run: str | None = None,
    spawn: bool = False,
) -> None:
    # GPU is set at registration time via DEFAULT_GPU (env MOEFORGE_GPU); A10G by default.
    target = run_smollm_recovery
    # Resume: start from a prior run's recovered-wrapper instead of the source carved wrapper.
    # That reloads the trained expert + router weights, so training continues rather than
    # restarting from scratch. (Optimizer momentum still resets; the flat LR makes that minor.)
    # Pass --train-experts again for a joint resume, or it will only continue training the router.
    if resume_from_run:
        wrapper = f"{REMOTE_ROOT_DISPLAY}/recovery-runs/{resume_from_run}/recovered-wrapper"
    kwargs = {
        "run_name": run_name,
        "wrapper": wrapper,
        "source_model": source_model,
        "steps": steps,
        "batch_size": batch_size,
        "sequence_length": sequence_length,
        "learning_rate": learning_rate,
        "train_experts": train_experts,
        "corpus_sources": corpus_sources,
        "max_samples_per_source": max_samples_per_source,
        "include_answers": include_answers,
        "token_router_top_k": token_router_top_k,
        "router_oracle_method": router_oracle_method,
        "eval_max_samples": eval_max_samples,
    }
    if spawn:
        call = target.spawn(**kwargs)
        manifest = {
            "format": "moeforge_modal_recovery_spawn",
            "run_name": run_name,
            "run_dir": f"{REMOTE_ROOT_DISPLAY}/recovery-runs/{run_name}",
            "function_call_id": call.object_id,
            "dashboard_url": call.get_dashboard_url(),
            "expected_report": f"{REMOTE_ROOT_DISPLAY}/recovery-runs/{run_name}/modal-recovery-manifest.json",
            "resume_from_run": resume_from_run,
            "wrapper": wrapper,
        }
    else:
        manifest = target.remote(**kwargs)
        if isinstance(manifest, dict):
            manifest["resume_from_run"] = resume_from_run
    print(json.dumps(manifest, indent=2, sort_keys=True))
