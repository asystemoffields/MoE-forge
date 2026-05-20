from __future__ import annotations

import json
from pathlib import Path

import modal


APP_NAME = "moeforge-smollm-recovery"
VOLUME_NAME = "moeforge-benchmarks"
MOEFORGE_REVISION = "3644bf59af2dc4d4b783e9ecffac622533a3e79f"
REMOTE_ROOT = Path("/vol")

TRAIN_TEXT = """
The quick brown fox jumps over the lazy dog. A scientist writes a hypothesis, collects evidence, and revises the claim.

Question: Which object is used to write on a chalkboard? A. spoon B. chalk C. blanket D. river Answer: B.

Question: If rain falls all night, what is likely to be wet in the morning? A. sidewalk B. flame C. desert map D. candle Answer: A.

Paris is the capital city of France. Water freezes when it gets cold enough. Plants often need sunlight, water, and soil.

Question: Which animal is known for laying eggs? A. chicken B. chair C. piano D. mountain Answer: A.

In a library, books are organized so readers can find them. A recipe lists ingredients and steps for cooking food.

Question: What tool is best for cutting paper? A. scissors B. pillow C. cloud D. bottle Answer: A.

The moon orbits Earth. A thermometer measures temperature. A map helps a traveler understand locations and routes.

Question: If a cup is full of hot tea, what should you do before drinking quickly? A. let it cool B. freeze the sun C. fold a stone D. erase the cup Answer: A.
""".strip()


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
) -> dict[str, object]:
    from moeforge.recovery_experiment import run_recovery_experiment

    run_dir = REMOTE_ROOT / "recovery-runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    train_file = run_dir / "train.txt"
    train_file.write_text(TRAIN_TEXT + "\n", encoding="utf-8")
    config = {
        "model": source_model,
        "wrapper": wrapper,
        "output_dir": str(run_dir),
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
        "summary": report.get("summary"),
        "artifacts": report.get("artifacts"),
    }


@app.local_entrypoint()
def main(
    run_name: str = "smollm-st-router-200",
    wrapper: str = "/vol/smollm-moe-v5",
    source_model: str = "/vol/smollm-moe-v5/source-model",
    steps: int = 200,
    batch_size: int = 1,
    sequence_length: int = 128,
    learning_rate: float = 1e-4,
    train_experts: bool = False,
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
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
