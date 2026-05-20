from __future__ import annotations

import json
from pathlib import Path
import shlex
import subprocess
from typing import Literal

import modal


APP_NAME = "moeforge-smollm-benchmark"
VOLUME_NAME = "moeforge-benchmarks"
LIGHTEVAL_REVISION = "v0.10.0"
REMOTE_ROOT = Path("/vol")
LIGHTEVAL_ROOT = Path("/opt/lighteval")
CUSTOM_TASKS = Path("/opt/lighteval_tasks.py")


app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "curl")
    .pip_install(
        "torch>=2.2",
        "transformers>=4.45",
        "safetensors>=0.4",
        "accelerate>=0.34",
        "datasets",
        "sentencepiece",
        "protobuf",
        "pillow",
    )
    .run_commands(
        f"git clone --depth 1 --branch {LIGHTEVAL_REVISION} https://github.com/huggingface/lighteval.git /opt/lighteval",
        "cd /opt/lighteval && pip install '.[accelerate,quantization,adapters]'",
        "curl -L https://raw.githubusercontent.com/huggingface/cosmopedia/main/evaluation/lighteval_tasks.py -o /opt/lighteval_tasks.py",
        "pip install git+https://github.com/asystemoffields/MoE-forge.git",
    )
)


@app.function(image=image, gpu="T4", timeout=60 * 60 * 12, volumes={str(REMOTE_ROOT): volume})
def run_benchmark_plan(
    plan_json: str,
    *,
    run_name: str,
    which: Literal["dense", "moe", "both"],
    source_model: str | None,
    moe_model: str | None,
    max_samples: int | None,
    batch_size: int | None,
) -> dict[str, object]:
    plan = json.loads(plan_json)
    if not isinstance(plan, dict):
        raise ValueError("benchmark plan must be a JSON object")
    suite = str(plan.get("suite") or "smollm-base")
    task_spec = str(plan["task_spec"])
    source = source_model or str(plan["source_model"])
    moe = moe_model or str(plan["moe_model"])
    sample_cap = int(max_samples or plan.get("max_samples") or 1000)
    effective_batch_size = int(batch_size or plan.get("batch_size") or 16)
    use_chat_template = "--use_chat_template" in str(plan.get("commands", {}).get("dense", ""))

    run_dir = REMOTE_ROOT / "runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "benchmark-plan.json").write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    runs: list[dict[str, object]] = []
    if which in {"dense", "both"}:
        runs.append(
            _run_lighteval(
                model=source,
                output_dir=run_dir / "dense",
                task_spec=task_spec,
                max_samples=sample_cap,
                batch_size=effective_batch_size,
                trust_remote_code=False,
                use_chat_template=use_chat_template,
            )
        )
    if which in {"moe", "both"}:
        runs.append(
            _run_lighteval(
                model=moe,
                output_dir=run_dir / "moe",
                task_spec=task_spec,
                max_samples=sample_cap,
                batch_size=effective_batch_size,
                trust_remote_code=True,
                use_chat_template=use_chat_template,
            )
        )
    manifest = {
        "format": "moeforge_modal_benchmark_manifest",
        "suite": suite,
        "run_name": run_name,
        "run_dir": str(run_dir),
        "which": which,
        "source_model": source,
        "moe_model": moe,
        "max_samples": sample_cap,
        "batch_size": effective_batch_size,
        "use_chat_template": use_chat_template,
        "runs": runs,
    }
    (run_dir / "modal-benchmark-manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    volume.commit()
    return manifest


def _run_lighteval(
    *,
    model: str,
    output_dir: Path,
    task_spec: str,
    max_samples: int,
    batch_size: int,
    trust_remote_code: bool,
    use_chat_template: bool,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    model_args = f"model_name={model},batch_size={batch_size}"
    if trust_remote_code:
        model_args += ",trust_remote_code=True"
    command = [
        "python",
        "-m",
        "lighteval",
        "accelerate",
        model_args,
        task_spec,
        "--custom-tasks",
        str(CUSTOM_TASKS),
        "--output-dir",
        str(output_dir),
        "--max-samples",
        str(max_samples),
        "--save-details",
    ]
    if use_chat_template:
        command.append("--use-chat-template")
    completed = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    (output_dir / "command.txt").write_text(shlex.join(command) + "\n", encoding="utf-8")
    (output_dir / "stdout.txt").write_text(completed.stdout, encoding="utf-8")
    json_outputs = [str(path) for path in output_dir.rglob("*.json")]
    canonical_results = _write_canonical_results(output_dir)
    return {
        "model": model,
        "output_dir": str(output_dir),
        "returncode": completed.returncode,
        "json_outputs": json_outputs,
        "canonical_results": str(canonical_results) if canonical_results else None,
        "stdout_path": str(output_dir / "stdout.txt"),
    }


def _write_canonical_results(output_dir: Path) -> Path | None:
    for path in sorted(output_dir.rglob("*.json")):
        if path.name == "results.json":
            return path
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict) and "results" in payload:
            canonical = output_dir / "results.json"
            canonical.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            return canonical
    return None


@app.local_entrypoint()
def main(
    plan: str,
    run_name: str = "smollm-benchmark",
    which: str = "both",
    source_model: str | None = None,
    moe_model: str | None = None,
    max_samples: int | None = None,
    batch_size: int | None = None,
) -> None:
    if which not in {"dense", "moe", "both"}:
        raise ValueError("--which must be dense, moe, or both")
    plan_json = Path(plan).read_text(encoding="utf-8")
    manifest = run_benchmark_plan.remote(
        plan_json,
        run_name=run_name,
        which=which,  # type: ignore[arg-type]
        source_model=source_model,
        moe_model=moe_model,
        max_samples=max_samples,
        batch_size=batch_size,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
