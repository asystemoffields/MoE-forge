from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .adapters import ADAPTERS
from .batch import run_eval_batch
from .benchmark import (
    BenchmarkCompareOptions,
    BenchmarkPlanOptions,
    compare_benchmark_reports,
    write_benchmark_plan,
)
from .carve import build_carve_manifest
from .conversion import ConversionRunOptions, run_conversion
from .evaluation import evaluate_hf_dense_vs_carved
from .inspectors import inspect_model
from .materialize import materialize_carve_manifest
from .model_card import write_model_card
from .planner import PlanOptions, plan_conversion
from .preflight import run_preflight
from .publish import check_publish_readiness
from .profiling import ProfileOptions, load_calibration_texts, profile_hf_model
from .recovery_compare import write_recovery_comparison_report
from .reports import (
    write_eval_comparison_report,
    write_eval_html_report,
    write_eval_html_report_payload,
)
from .recovery import write_recovery_plan
from .recovery_experiment import run_recovery_experiment
from .recovery_runner import export_recovered_wrapper, run_recovery, validate_recovered_wrapper
from .recipe import recipe_to_dict
from .router import build_router_plan
from .runtime import verify_carved_artifact
from .smoke import assert_tiny_hf_smoke_run
from .wrapper import export_wrapper_package


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "inspect":
            return _cmd_inspect(args)
        if args.command == "plan":
            return _cmd_plan(args)
        if args.command == "adapters":
            return _cmd_adapters(args)
        if args.command == "preflight":
            return _cmd_preflight(args)
        if args.command == "convert":
            return _cmd_convert(args)
        if args.command == "publish-check":
            return _cmd_publish_check(args)
        if args.command == "benchmark-plan":
            return _cmd_benchmark_plan(args)
        if args.command == "benchmark-compare":
            return _cmd_benchmark_compare(args)
        if args.command == "profile":
            return _cmd_profile(args)
        if args.command == "carve-manifest":
            return _cmd_carve_manifest(args)
        if args.command == "carve-apply":
            return _cmd_carve_apply(args)
        if args.command == "carve-verify":
            return _cmd_carve_verify(args)
        if args.command == "router-plan":
            return _cmd_router_plan(args)
        if args.command == "wrapper-export":
            return _cmd_wrapper_export(args)
        if args.command == "eval-hf":
            return _cmd_eval_hf(args)
        if args.command == "eval-report-html":
            return _cmd_eval_report_html(args)
        if args.command == "eval-compare":
            return _cmd_eval_compare(args)
        if args.command == "eval-batch":
            return _cmd_eval_batch(args)
        if args.command == "recovery-plan":
            return _cmd_recovery_plan(args)
        if args.command == "recovery-run":
            return _cmd_recovery_run(args)
        if args.command == "recovery-export":
            return _cmd_recovery_export(args)
        if args.command == "recovery-validate":
            return _cmd_recovery_validate(args)
        if args.command == "recovery-experiment":
            return _cmd_recovery_experiment(args)
        if args.command == "recovery-compare":
            return _cmd_recovery_compare(args)
        if args.command == "model-card":
            return _cmd_model_card(args)
        if args.command == "smoke-assert":
            return _cmd_smoke_assert(args)
    except Exception as exc:  # pragma: no cover - CLI boundary
        print(f"error: {exc}", file=sys.stderr)
        return 2

    parser.print_help()
    return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="moe-forge",
        description="Plan and build Mixture-of-Experts variants from dense model checkpoints.",
    )
    subparsers = parser.add_subparsers(dest="command")

    adapters_parser = subparsers.add_parser("adapters", help="List supported architecture adapters.")
    adapters_parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")

    inspect_parser = subparsers.add_parser("inspect", help="Inspect a local HF model folder or GGUF file.")
    inspect_parser.add_argument("model", help="Path to a model folder, config.json, GGUF file, or HF model id.")
    inspect_parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    inspect_parser.add_argument("--output", type=Path, help="Write inspection JSON to this path.")

    preflight_parser = subparsers.add_parser(
        "preflight",
        help="Check dense-to-MoE workflow readiness and suggest next agent-safe commands.",
    )
    preflight_parser.add_argument("--model", help="Dense HF model folder, config path, GGUF file, or HF model id.")
    preflight_parser.add_argument("--recipe", type=Path, help="Conversion recipe JSON.")
    preflight_parser.add_argument("--profile", type=Path, help="Activation profile JSON.")
    preflight_parser.add_argument("--manifest", type=Path, help="Carve manifest JSON.")
    preflight_parser.add_argument("--artifact", type=Path, help="Carved safetensors artifact.")
    preflight_parser.add_argument("--wrapper", type=Path, help="Wrapper package directory.")
    preflight_parser.add_argument("--recovery-config", type=Path, help="Recovery or recovery-experiment JSON config.")
    preflight_parser.add_argument("--output", type=Path, default=Path("preflight-report.json"), help="Preflight JSON output path.")
    preflight_parser.add_argument("--print", action="store_true", help="Also print the preflight report JSON.")

    convert_parser = subparsers.add_parser(
        "convert",
        help="Run a dense-to-MoE conversion into a native HF wrapper package.",
    )
    convert_parser.add_argument("model", help="Path to a local HF model folder.")
    convert_parser.add_argument("--output-dir", type=Path, required=True, help="Run directory for recipe, tensors, wrapper, and reports.")
    convert_parser.add_argument("--recipe", type=Path, help="Existing conversion recipe JSON. Defaults to planning one.")
    convert_parser.add_argument("--profile", type=Path, help="Optional activation profile JSON for channel assignment.")
    convert_parser.add_argument(
        "--goal",
        choices=["balanced", "speed", "quality", "tiny", "explore"],
        default="balanced",
        help="High-level optimization target when planning a recipe.",
    )
    convert_parser.add_argument("--target", choices=["hf", "gguf", "analysis"], default="hf", help="Preferred output target.")
    convert_parser.add_argument("--hardware", default="auto", help="Hardware hint such as cpu, laptop, cuda, or auto.")
    convert_parser.add_argument("--experts", type=int, help="Number of routed experts per converted layer.")
    convert_parser.add_argument("--top-k", type=int, help="Number of routed experts active per token.")
    convert_parser.add_argument("--shared-ratio", type=float, help="Fraction of FFN channels reserved for shared path.")
    convert_parser.add_argument("--moe-layers", default="all", help="Layer range, list, or all. Defaults to all.")
    convert_parser.add_argument("--calibration-samples", type=int, help="Calibration sample count for planned recipes.")
    convert_parser.add_argument("--recover-steps", type=int, help="Recovery step count for planned recipes.")
    convert_parser.add_argument("--activation", default="silu", help="FFN activation: silu, gelu, or gelu_tanh.")
    convert_parser.add_argument("--token-router-top-k", type=int, help="Enable learned per-token top-k routing in the package.")
    convert_parser.add_argument("--skip-source-model-copy", action="store_true", help="Reference the dense model path instead of copying it into the wrapper.")
    convert_parser.add_argument("--dry-run", action="store_true", help="Write plan, manifest, preflight, and tensor-shape reports without exporting a wrapper.")
    convert_parser.add_argument("--eval-smoke", action="store_true", help="Run deterministic dense-vs-carved smoke eval after wrapper export.")
    convert_parser.add_argument(
        "--eval-expert-mode",
        action="append",
        choices=["all", "default-pool", "router", "learned-router"],
        help="Expert mode for --eval-smoke. Can be supplied multiple times.",
    )
    convert_parser.add_argument("--eval-device", default="cpu", help="Torch device for --eval-smoke.")
    convert_parser.add_argument("--eval-sequence-length", type=int, default=128, help="Generated smoke input length cap.")
    convert_parser.add_argument("--eval-atol", type=float, default=1e-5, help="Absolute allclose tolerance for smoke eval.")
    convert_parser.add_argument("--eval-rtol", type=float, default=1e-5, help="Relative allclose tolerance for smoke eval.")
    convert_parser.add_argument("--recover", action="store_true", help="Run recovery training, export a recovered wrapper, and publish-check it.")
    convert_parser.add_argument("--recover-experts", action="store_true", help="Also train carved expert tensors during recovery.")
    convert_parser.add_argument("--train-text", help="Inline training text for recovery.")
    convert_parser.add_argument("--train-text-file", type=Path, help="Training text file for recovery; blank lines split samples.")
    convert_parser.add_argument("--train-input-ids-json", help="Recovery train token ids JSON, such as [[1,2,3]].")
    convert_parser.add_argument("--eval-text", help="Inline eval text for recovery and publish evidence.")
    convert_parser.add_argument("--eval-text-file", type=Path, help="Eval text file; blank lines split samples.")
    convert_parser.add_argument("--eval-input-ids-json", help="Recovery eval token ids JSON, such as [[1,2,3]].")
    convert_parser.add_argument("--max-all-expert-error", type=float, default=1e-4, help="Publish-check max all-expert reconstruction error.")
    convert_parser.add_argument("--max-all-expert-teacher-kl", type=float, default=0.01, help="Publish-check max all-expert teacher-KL fallback.")
    convert_parser.add_argument("--max-sparse-teacher-kl", type=float, help="Publish-check max sparse teacher-KL.")
    convert_parser.add_argument("--max-sparse-nll-delta", type=float, help="Publish-check max sparse NLL delta.")
    convert_parser.add_argument("--print", action="store_true", help="Also print the conversion report JSON.")

    publish_parser = subparsers.add_parser(
        "publish-check",
        help="Check whether a wrapper package has the evidence expected for HF publication.",
    )
    publish_parser.add_argument("--wrapper", type=Path, required=True, help="Wrapper or recovered-wrapper package directory.")
    publish_parser.add_argument("--eval-report", type=Path, action="append", default=[], help="Eval JSON report. Can be supplied multiple times.")
    publish_parser.add_argument("--benchmark-report", type=Path, help="Benchmark comparison JSON from benchmark-compare.")
    publish_parser.add_argument("--recovery-report", type=Path, help="Recovery experiment JSON report.")
    publish_parser.add_argument("--validation-report", type=Path, help="Recovered-wrapper validation JSON report.")
    publish_parser.add_argument("--require-recovery", action="store_true", help="Require recovery and validation evidence.")
    publish_parser.add_argument("--require-benchmark", action="store_true", help="Require source-aligned benchmark comparison evidence.")
    publish_parser.add_argument("--allow-missing-sparse-eval", action="store_true", help="Do not block when sparse eval reports are absent.")
    publish_parser.add_argument("--max-all-expert-error", type=float, default=1e-4, help="Max all-expert reconstruction error.")
    publish_parser.add_argument("--max-all-expert-teacher-kl", type=float, default=0.01, help="Max all-expert teacher-KL fallback.")
    publish_parser.add_argument("--max-sparse-teacher-kl", type=float, help="Max sparse teacher-KL.")
    publish_parser.add_argument("--max-sparse-nll-delta", type=float, help="Max sparse NLL delta.")
    publish_parser.add_argument("--skip-native-load", action="store_true", help="Skip native AutoModel load check.")
    publish_parser.add_argument("--output", type=Path, default=Path("publish-readiness.json"), help="Publish readiness JSON output path.")
    publish_parser.add_argument("--print", action="store_true", help="Also print the publish readiness report JSON.")

    benchmark_plan_parser = subparsers.add_parser(
        "benchmark-plan",
        help="Write a reproducible dense-vs-MoE benchmark plan for a source checkpoint.",
    )
    benchmark_plan_parser.add_argument("--source-model", required=True, help="Dense source model path or HF model id.")
    benchmark_plan_parser.add_argument("--moe-model", required=True, help="MoE wrapper path or HF model id.")
    benchmark_plan_parser.add_argument(
        "--suite",
        choices=["smollm-base", "smollm-instruct"],
        default="smollm-base",
        help="Benchmark suite matched to the source checkpoint family.",
    )
    benchmark_plan_parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark-plan.json"),
        help="Benchmark plan JSON output path.",
    )
    benchmark_plan_parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory expected to hold raw benchmark outputs.",
    )
    benchmark_plan_parser.add_argument("--max-samples", type=int, default=1000, help="Max samples per task.")
    benchmark_plan_parser.add_argument("--batch-size", type=int, default=16, help="Harness batch size.")
    benchmark_plan_parser.add_argument(
        "--custom-tasks-path",
        default="lighteval_tasks.py",
        help="Path to the SmolLM/Cosmopedia LightEval custom tasks file.",
    )
    benchmark_plan_parser.add_argument("--no-gsm8k", action="store_true", help="Omit GSM8K from the task list.")
    benchmark_plan_parser.add_argument("--use-chat-template", action="store_true", help="Use chat templates for both dense and MoE runs.")
    benchmark_plan_parser.add_argument("--print", action="store_true", help="Also print the benchmark plan JSON.")

    benchmark_compare_parser = subparsers.add_parser(
        "benchmark-compare",
        help="Compare dense and MoE benchmark JSON outputs and apply a release-quality gate.",
    )
    benchmark_compare_parser.add_argument("--dense-report", type=Path, required=True, help="Dense benchmark result JSON.")
    benchmark_compare_parser.add_argument("--moe-report", type=Path, required=True, help="MoE benchmark result JSON.")
    benchmark_compare_parser.add_argument(
        "--suite",
        choices=["smollm-base", "smollm-instruct"],
        default="smollm-base",
        help="Benchmark suite used for the run.",
    )
    benchmark_compare_parser.add_argument("--output", type=Path, required=True, help="Comparison JSON output path.")
    benchmark_compare_parser.add_argument("--max-average-drop", type=float, default=0.05, help="Max allowed average absolute score drop.")
    benchmark_compare_parser.add_argument("--max-core-drop", type=float, default=0.08, help="Max allowed single core-task score drop.")
    benchmark_compare_parser.add_argument("--min-average-retention", type=float, default=0.95, help="Min allowed average score retention.")
    benchmark_compare_parser.add_argument("--min-core-retention", type=float, default=0.90, help="Min allowed worst core-task retention.")
    benchmark_compare_parser.add_argument("--print", action="store_true", help="Also print the comparison JSON.")

    plan_parser = subparsers.add_parser("plan", help="Create a dense-to-MoE conversion recipe.")
    plan_parser.add_argument("model", help="Path to a model folder, config.json, GGUF file, or HF model id.")
    plan_parser.add_argument(
        "--goal",
        choices=["balanced", "speed", "quality", "tiny", "explore"],
        default="balanced",
        help="High-level optimization target.",
    )
    plan_parser.add_argument(
        "--target",
        choices=["hf", "gguf", "analysis"],
        default="hf",
        help="Preferred output target.",
    )
    plan_parser.add_argument("--hardware", default="auto", help="Hardware hint such as cpu, laptop, cuda, or auto.")
    plan_parser.add_argument("--experts", type=int, help="Number of routed experts per converted layer.")
    plan_parser.add_argument("--top-k", type=int, help="Number of routed experts active per token.")
    plan_parser.add_argument("--shared-ratio", type=float, help="Fraction of FFN channels reserved for shared path.")
    plan_parser.add_argument("--moe-layers", help="Layer range, list, or all, such as 8:34, 8,10,12, or all.")
    plan_parser.add_argument("--calibration-samples", type=int, help="Calibration text sample count.")
    plan_parser.add_argument("--recover-steps", type=int, help="Recovery training step count.")
    plan_parser.add_argument("--output", type=Path, default=Path("recipe.json"), help="Recipe output path.")
    plan_parser.add_argument("--print", action="store_true", help="Also print the recipe JSON.")

    profile_parser = subparsers.add_parser("profile", help="Profile FFN activations on calibration text.")
    profile_parser.add_argument("model", help="Path to a local HF model folder or HF model id.")
    profile_parser.add_argument("--text", help="Inline calibration text sample.")
    profile_parser.add_argument("--text-file", type=Path, help="Calibration text file; blank lines split samples.")
    profile_parser.add_argument("--layers", help="Layer range, list, or all, such as 8:34, 8,10,12, or all.")
    profile_parser.add_argument("--roles", default="gate,up", help="Comma-separated FFN roles to hook: gate,up,down.")
    profile_parser.add_argument("--max-samples", type=int, default=32, help="Maximum calibration samples to run.")
    profile_parser.add_argument("--sequence-length", type=int, default=512, help="Tokenizer truncation length.")
    profile_parser.add_argument("--device", default="auto", help="Torch device: auto, cpu, cuda, cuda:0, etc.")
    profile_parser.add_argument("--dtype", default="auto", help="Model dtype: auto, fp32, fp16, bf16.")
    profile_parser.add_argument("--threshold", type=float, default=0.0, help="Absolute activation threshold for active-rate stats.")
    profile_parser.add_argument("--top-k-channels", type=int, default=32, help="Top channels to include per module.")
    profile_parser.add_argument("--document-top-k-channels", type=int, default=8, help="Top channels to include per document/module.")
    profile_parser.add_argument("--experts", type=int, default=8, help="Experts per profiled FFN module for assignment suggestions.")
    profile_parser.add_argument("--shared-ratio", type=float, default=0.25, help="Shared-channel ratio for assignment suggestions.")
    profile_parser.add_argument("--include-vectors", action="store_true", help="Include full per-channel vectors in JSON.")
    profile_parser.add_argument("--include-document-vectors", action="store_true", help="Include full per-document per-channel vectors in JSON.")
    profile_parser.add_argument("--output", type=Path, default=Path("activation-profile.json"), help="Profile output path.")
    profile_parser.add_argument("--print", action="store_true", help="Also print the profile JSON.")

    carve_parser = subparsers.add_parser(
        "carve-manifest",
        help="Build a validated carve manifest from a recipe and optional activation profile.",
    )
    carve_parser.add_argument("model", help="Path to a local HF model folder.")
    carve_parser.add_argument("--recipe", type=Path, required=True, help="Recipe JSON from moe-forge plan.")
    carve_parser.add_argument("--profile", type=Path, help="Activation profile JSON from moe-forge profile.")
    carve_parser.add_argument("--output", type=Path, default=Path("carve-manifest.json"), help="Manifest output path.")
    carve_parser.add_argument("--print", action="store_true", help="Also print the manifest JSON.")

    apply_parser = subparsers.add_parser(
        "carve-apply",
        help="Materialize carved shared/expert tensors from a carve manifest.",
    )
    apply_parser.add_argument("--manifest", type=Path, required=True, help="Carve manifest JSON.")
    apply_parser.add_argument("--output-dir", type=Path, required=True, help="Directory for carved safetensors and report.")
    apply_parser.add_argument("--dry-run", action="store_true", help="Validate and report planned tensor outputs without writing safetensors.")
    apply_parser.add_argument("--print", action="store_true", help="Also print the materialization report JSON.")

    verify_parser = subparsers.add_parser(
        "carve-verify",
        help="Verify carved tensors reconstruct the source dense FFN weights.",
    )
    verify_parser.add_argument("--manifest", type=Path, required=True, help="Carve manifest JSON.")
    verify_parser.add_argument("--artifact", type=Path, required=True, help="carved-experts.safetensors path.")
    verify_parser.add_argument("--atol", type=float, default=1e-6, help="Absolute allclose tolerance.")
    verify_parser.add_argument("--rtol", type=float, default=1e-5, help="Relative allclose tolerance.")
    verify_parser.add_argument("--output", type=Path, default=Path("carve-verify-report.json"), help="Verification report output path.")
    verify_parser.add_argument("--print", action="store_true", help="Also print the verification report JSON.")

    router_parser = subparsers.add_parser(
        "router-plan",
        help="Build EMO-style document expert-pool router metadata from a profile report.",
    )
    router_parser.add_argument("--profile", type=Path, required=True, help="Profile JSON from moe-forge profile.")
    router_parser.add_argument("--pool-size", type=int, help="Experts to keep per document.")
    router_parser.add_argument("--output", type=Path, default=Path("router-plan.json"), help="Router plan output path.")
    router_parser.add_argument("--print", action="store_true", help="Also print the router plan JSON.")

    wrapper_parser = subparsers.add_parser(
        "wrapper-export",
        help="Export a runnable MoE Forge wrapper package for carved FFN artifacts.",
    )
    wrapper_parser.add_argument("--manifest", type=Path, required=True, help="Carve manifest JSON.")
    wrapper_parser.add_argument("--artifact", type=Path, required=True, help="carved-experts.safetensors path.")
    wrapper_parser.add_argument("--router-plan", type=Path, help="Optional router-plan JSON.")
    wrapper_parser.add_argument("--activation", default="silu", help="FFN activation: silu, gelu, or gelu_tanh.")
    wrapper_parser.add_argument("--copy-artifact", action="store_true", help="Copy the safetensors artifact into the wrapper directory.")
    wrapper_parser.add_argument("--copy-source-model", action="store_true", help="Copy the dense source checkpoint into the wrapper package.")
    wrapper_parser.add_argument("--token-router-top-k", type=int, help="Enable learned per-token top-k routing in the wrapper package.")
    wrapper_parser.add_argument("--output-dir", type=Path, required=True, help="Wrapper package output directory.")
    wrapper_parser.add_argument("--print", action="store_true", help="Also print the wrapper config JSON.")

    eval_parser = subparsers.add_parser(
        "eval-hf",
        help="Evaluate dense-vs-carved HF parity with a wrapper package.",
    )
    eval_parser.add_argument("model", help="Path to a local HF model folder.")
    eval_parser.add_argument("--wrapper", type=Path, required=True, help="MoE Forge wrapper package directory.")
    eval_parser.add_argument("--text", help="Inline evaluation text sample.")
    eval_parser.add_argument("--text-file", type=Path, help="Evaluation text file; blank lines split samples.")
    eval_parser.add_argument("--input-ids-json", help="JSON list of token id lists, such as [[1,2,3]].")
    eval_parser.add_argument("--sequence-length", type=int, default=128, help="Tokenizer truncation length or generated smoke input length.")
    eval_parser.add_argument("--device", default="cpu", help="Torch device: cpu, auto, cuda, cuda:0, etc.")
    eval_parser.add_argument(
        "--expert-mode",
        choices=["all", "default-pool", "router", "learned-router"],
        default="all",
        help="Experts active in carved FFNs during evaluation.",
    )
    eval_parser.add_argument("--atol", type=float, default=1e-5, help="Absolute allclose tolerance for logits.")
    eval_parser.add_argument("--rtol", type=float, default=1e-5, help="Relative allclose tolerance for logits.")
    eval_parser.add_argument("--strict", action="store_true", help="Return non-zero when logits do not pass allclose.")
    eval_parser.add_argument("--output", type=Path, default=Path("moeforge-eval-report.json"), help="Evaluation report output path.")
    eval_parser.add_argument("--html-output", type=Path, help="Optional self-contained HTML report output path.")
    eval_parser.add_argument("--print", action="store_true", help="Also print the evaluation report JSON.")

    report_parser = subparsers.add_parser(
        "eval-report-html",
        help="Render an eval-hf JSON report as self-contained HTML.",
    )
    report_parser.add_argument("--input", type=Path, required=True, help="Evaluation JSON report from eval-hf.")
    report_parser.add_argument("--output", type=Path, required=True, help="HTML report output path.")

    compare_parser = subparsers.add_parser(
        "eval-compare",
        help="Compare multiple eval-hf JSON reports side by side.",
    )
    compare_parser.add_argument("reports", type=Path, nargs="+", help="Evaluation JSON reports from eval-hf.")
    compare_parser.add_argument("--output", type=Path, required=True, help="Comparison JSON output path.")
    compare_parser.add_argument("--html-output", type=Path, help="Optional self-contained HTML comparison output path.")
    compare_parser.add_argument("--print", action="store_true", help="Also print the comparison JSON.")

    batch_parser = subparsers.add_parser(
        "eval-batch",
        help="Run multiple eval-hf modes from a JSON batch config.",
    )
    batch_parser.add_argument("--config", type=Path, required=True, help="Eval batch JSON config.")
    batch_parser.add_argument("--output-dir", type=Path, help="Override the config output_dir.")
    batch_parser.add_argument("--strict", action="store_true", help="Return non-zero when any completed mode fails.")
    batch_parser.add_argument("--print", action="store_true", help="Also print the batch manifest JSON.")

    recovery_parser = subparsers.add_parser(
        "recovery-plan",
        help="Build a teacher-KL recovery-training plan artifact.",
    )
    recovery_parser.add_argument("--config", type=Path, required=True, help="Recovery plan JSON config.")
    recovery_parser.add_argument("--output", type=Path, help="Recovery plan output path.")
    recovery_parser.add_argument("--print", action="store_true", help="Also print the recovery plan JSON.")

    recovery_run_parser = subparsers.add_parser(
        "recovery-run",
        help="Run a tiny teacher-KL recovery loop from a recovery plan.",
    )
    recovery_run_parser.add_argument("--plan", type=Path, required=True, help="Recovery plan JSON artifact.")
    recovery_run_parser.add_argument("--output", type=Path, help="Recovery run report output path.")
    recovery_run_parser.add_argument("--max-steps", type=int, help="Override planned step count for smoke runs.")
    recovery_run_parser.add_argument("--print", action="store_true", help="Also print the recovery run report JSON.")

    recovery_export_parser = subparsers.add_parser(
        "recovery-export",
        help="Apply a recovery checkpoint to a wrapper artifact.",
    )
    recovery_export_parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Recovery checkpoint metadata JSON.",
    )
    recovery_export_parser.add_argument("--wrapper", type=Path, required=True, help="Source wrapper package directory.")
    recovery_export_parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Recovered wrapper output directory.",
    )
    recovery_export_parser.add_argument(
        "--artifact-name",
        default="recovered-carved-experts.safetensors",
        help="Recovered artifact filename.",
    )
    recovery_export_parser.add_argument(
        "--print",
        action="store_true",
        help="Also print the recovery export report JSON.",
    )

    recovery_validate_parser = subparsers.add_parser(
        "recovery-validate",
        help="Validate a recovered wrapper package against its source wrapper and checkpoint.",
    )
    recovery_validate_parser.add_argument("--source-wrapper", type=Path, required=True, help="Original wrapper package directory.")
    recovery_validate_parser.add_argument("--recovered-wrapper", type=Path, required=True, help="Recovered wrapper package directory.")
    recovery_validate_parser.add_argument("--checkpoint", type=Path, help="Recovery checkpoint metadata JSON.")
    recovery_validate_parser.add_argument("--export-report", type=Path, help="Recovery export report JSON.")
    recovery_validate_parser.add_argument(
        "--output",
        type=Path,
        default=Path("recovered-wrapper-validation.json"),
        help="Validation report output path.",
    )
    recovery_validate_parser.add_argument(
        "--print",
        action="store_true",
        help="Also print the validation report JSON.",
    )

    recovery_experiment_parser = subparsers.add_parser(
        "recovery-experiment",
        help="Run before/after eval, recovery, export, and comparison from one config.",
    )
    recovery_experiment_parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Recovery experiment JSON config.",
    )
    recovery_experiment_parser.add_argument("--output-dir", type=Path, help="Experiment output directory.")
    recovery_experiment_parser.add_argument("--max-steps", type=int, help="Override planned recovery step count.")
    recovery_experiment_parser.add_argument(
        "--print",
        action="store_true",
        help="Also print the experiment report JSON.",
    )

    recovery_compare_parser = subparsers.add_parser(
        "recovery-compare",
        help="Compare multiple recovery-experiment JSON reports side by side.",
    )
    recovery_compare_parser.add_argument(
        "reports",
        type=Path,
        nargs="+",
        help="Recovery experiment JSON reports.",
    )
    recovery_compare_parser.add_argument("--output", type=Path, required=True, help="Comparison JSON output path.")
    recovery_compare_parser.add_argument("--html-output", type=Path, help="Optional self-contained HTML output path.")
    recovery_compare_parser.add_argument("--print", action="store_true", help="Also print the comparison JSON.")

    model_card_parser = subparsers.add_parser(
        "model-card",
        help="Write a package-ready Markdown model card from wrapper and report artifacts.",
    )
    model_card_parser.add_argument("--wrapper", type=Path, required=True, help="Wrapper package directory.")
    model_card_parser.add_argument(
        "--output",
        type=Path,
        default=Path("MODEL_CARD.md"),
        help="Markdown model-card output path.",
    )
    model_card_parser.add_argument(
        "--eval-report",
        type=Path,
        action="append",
        default=[],
        help="Evaluation JSON report to summarize. Can be supplied multiple times.",
    )
    model_card_parser.add_argument(
        "--recovery-report",
        type=Path,
        action="append",
        default=[],
        help="Recovery or recovery-experiment JSON report to summarize. Can be supplied multiple times.",
    )
    model_card_parser.add_argument(
        "--validation-report",
        type=Path,
        action="append",
        default=[],
        help="Recovered-wrapper validation JSON report to summarize. Can be supplied multiple times.",
    )
    model_card_parser.add_argument(
        "--command",
        dest="commands",
        action="append",
        default=[],
        help="Reproduction command to include in the card. Can be supplied multiple times.",
    )
    model_card_parser.add_argument("--print", action="store_true", help="Also print the model-card summary JSON.")

    smoke_parser = subparsers.add_parser(
        "smoke-assert",
        help="Assert expected artifacts and metrics from a tiny HF smoke run.",
    )
    smoke_parser.add_argument("--run-dir", type=Path, default=Path("."), help="Directory containing the smoke run artifacts.")
    smoke_parser.add_argument(
        "--output",
        type=Path,
        help="Smoke assertion JSON output path. Defaults to <run-dir>/smoke-assertions.json.",
    )
    smoke_parser.add_argument("--print", action="store_true", help="Also print the assertion report JSON.")

    return parser


def _cmd_inspect(args: argparse.Namespace) -> int:
    info = inspect_model(args.model)
    payload = info.to_dict()
    if args.output:
        _write_json(args.output, payload)
    if args.json or not args.output:
        print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _cmd_preflight(args: argparse.Namespace) -> int:
    report = run_preflight(
        model=args.model,
        recipe=args.recipe,
        profile=args.profile,
        manifest=args.manifest,
        artifact=args.artifact,
        wrapper=args.wrapper,
        recovery_config=args.recovery_config,
        output_path=args.output,
    )
    if args.print:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(f"{report['status']}; wrote {args.output}")
    return 0 if report.get("passed") else 1


def _cmd_convert(args: argparse.Namespace) -> int:
    report = run_conversion(
        ConversionRunOptions(
            model=args.model,
            output_dir=args.output_dir,
            recipe=args.recipe,
            profile=args.profile,
            goal=args.goal,
            target=args.target,
            hardware=args.hardware,
            experts=args.experts,
            top_k=args.top_k,
            shared_ratio=args.shared_ratio,
            moe_layers=args.moe_layers,
            calibration_samples=args.calibration_samples,
            recover_steps=args.recover_steps,
            activation=args.activation,
            token_router_top_k=args.token_router_top_k,
            copy_source_model=not args.skip_source_model_copy,
            dry_run=args.dry_run,
            eval_smoke=args.eval_smoke,
            eval_expert_modes=args.eval_expert_mode,
            eval_device=args.eval_device,
            eval_sequence_length=args.eval_sequence_length,
            eval_atol=args.eval_atol,
            eval_rtol=args.eval_rtol,
            recover=args.recover,
            recover_experts=args.recover_experts,
            train_text=args.train_text,
            train_text_file=args.train_text_file,
            train_input_ids_json=args.train_input_ids_json,
            eval_text=args.eval_text,
            eval_text_file=args.eval_text_file,
            eval_input_ids_json=args.eval_input_ids_json,
            max_all_expert_error=args.max_all_expert_error,
            max_all_expert_teacher_kl=args.max_all_expert_teacher_kl,
            max_sparse_teacher_kl=args.max_sparse_teacher_kl,
            max_sparse_nll_delta=args.max_sparse_nll_delta,
        )
    )
    if args.print:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(f"{report['status']}; wrote {report['artifacts']['convert_report']}")
    return 0 if report.get("passed") else 1


def _cmd_publish_check(args: argparse.Namespace) -> int:
    report = check_publish_readiness(
        wrapper=args.wrapper,
        output_path=args.output,
        eval_reports=args.eval_report,
        benchmark_report=args.benchmark_report,
        recovery_report=args.recovery_report,
        validation_report=args.validation_report,
        require_recovery=args.require_recovery,
        require_benchmark=args.require_benchmark,
        require_sparse_eval=not args.allow_missing_sparse_eval,
        max_all_expert_error=args.max_all_expert_error,
        max_all_expert_teacher_kl=args.max_all_expert_teacher_kl,
        max_sparse_teacher_kl=args.max_sparse_teacher_kl,
        max_sparse_nll_delta=args.max_sparse_nll_delta,
        trust_remote_code_load=not args.skip_native_load,
    )
    if args.print:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(f"{report['status']}; wrote {args.output}")
    return 0 if report.get("passed") else 1


def _cmd_benchmark_plan(args: argparse.Namespace) -> int:
    report = write_benchmark_plan(
        BenchmarkPlanOptions(
            source_model=args.source_model,
            moe_model=args.moe_model,
            output_path=args.output,
            suite=args.suite,
            output_dir=args.output_dir or Path("benchmarks") / args.suite,
            max_samples=args.max_samples,
            batch_size=args.batch_size,
            custom_tasks_path=args.custom_tasks_path,
            include_gsm8k=not args.no_gsm8k,
            use_chat_template=args.use_chat_template,
        )
    )
    if args.print:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(f"wrote {args.output}")
    return 0


def _cmd_benchmark_compare(args: argparse.Namespace) -> int:
    report = compare_benchmark_reports(
        BenchmarkCompareOptions(
            dense_report=args.dense_report,
            moe_report=args.moe_report,
            output_path=args.output,
            suite=args.suite,
            max_average_drop=args.max_average_drop,
            max_core_drop=args.max_core_drop,
            min_average_retention=args.min_average_retention,
            min_core_retention=args.min_core_retention,
        )
    )
    if args.print:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(f"{report['status']}; wrote {args.output}")
    return 0 if report.get("passed") else 1


def _cmd_plan(args: argparse.Namespace) -> int:
    info = inspect_model(args.model)
    options = PlanOptions(
        goal=args.goal,
        target=args.target,
        hardware=args.hardware,
        experts=args.experts,
        top_k=args.top_k,
        shared_ratio=args.shared_ratio,
        moe_layers=args.moe_layers,
        calibration_samples=args.calibration_samples,
        recover_steps=args.recover_steps,
    )
    recipe = plan_conversion(info, options)
    payload = recipe_to_dict(recipe)
    _write_json(args.output, payload)
    if args.print:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"wrote {args.output}")
    return 0


def _cmd_adapters(args: argparse.Namespace) -> int:
    payload = [adapter.to_dict() for adapter in ADAPTERS]
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    for adapter in ADAPTERS:
        backends = ", ".join(adapter.supported_backends)
        print(f"{adapter.family}: {adapter.ffn_kind}; backends: {backends}")
    return 0


def _cmd_profile(args: argparse.Namespace) -> int:
    roles = tuple(role.strip() for role in args.roles.split(",") if role.strip())
    texts = load_calibration_texts(
        text=args.text,
        text_file=args.text_file,
        max_samples=args.max_samples,
    )
    options = ProfileOptions(
        layers=args.layers,
        roles=roles,
        max_samples=args.max_samples,
        sequence_length=args.sequence_length,
        device=args.device,
        dtype=args.dtype,
        threshold=args.threshold,
        include_vectors=args.include_vectors,
        include_document_vectors=args.include_document_vectors,
        top_k_channels=args.top_k_channels,
        document_top_k_channels=args.document_top_k_channels,
        experts=args.experts,
        shared_ratio=args.shared_ratio,
    )
    payload = profile_hf_model(args.model, texts, options)
    _write_json(args.output, payload)
    if args.print:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"wrote {args.output}")
    return 0


def _cmd_carve_manifest(args: argparse.Namespace) -> int:
    manifest = build_carve_manifest(
        model=args.model,
        recipe_path=args.recipe,
        profile_path=args.profile,
    )
    payload = manifest.to_dict()
    _write_json(args.output, payload)
    if args.print:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"wrote {args.output}")
    return 0


def _cmd_carve_apply(args: argparse.Namespace) -> int:
    report = materialize_carve_manifest(
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        dry_run=args.dry_run,
    )
    payload = report.to_dict()
    if args.dry_run:
        if args.print:
            print(json.dumps(payload, indent=2, sort_keys=True))
        else:
            print(f"planned {len(report.tensors)} tensors; wrote {args.output_dir}")
        return 0
    if args.print:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"wrote {args.output_dir}")
    return 0


def _cmd_carve_verify(args: argparse.Namespace) -> int:
    report = verify_carved_artifact(
        manifest_path=args.manifest,
        artifact_path=args.artifact,
        atol=args.atol,
        rtol=args.rtol,
    )
    payload = report.to_dict()
    _write_json(args.output, payload)
    if args.print:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        status = "passed" if report.passed else "failed"
        print(f"{status}; wrote {args.output}")
    return 0 if report.passed else 1


def _cmd_router_plan(args: argparse.Namespace) -> int:
    plan = build_router_plan(profile_path=args.profile, pool_size=args.pool_size)
    payload = plan.to_dict()
    _write_json(args.output, payload)
    if args.print:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"wrote {args.output}")
    return 0


def _cmd_wrapper_export(args: argparse.Namespace) -> int:
    config = export_wrapper_package(
        manifest_path=args.manifest,
        artifact_path=args.artifact,
        output_dir=args.output_dir,
        router_plan_path=args.router_plan,
        activation=args.activation,
        copy_artifact=args.copy_artifact,
        copy_source_model=args.copy_source_model,
        token_router_top_k=args.token_router_top_k,
    )
    payload = config.to_dict()
    if args.print:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"wrote {args.output_dir}")
    return 0


def _cmd_eval_hf(args: argparse.Namespace) -> int:
    texts = _load_optional_texts(text=args.text, text_file=args.text_file)
    input_ids = json.loads(args.input_ids_json) if args.input_ids_json else None
    report = evaluate_hf_dense_vs_carved(
        model=args.model,
        package_dir=args.wrapper,
        texts=texts,
        input_ids=input_ids,
        sequence_length=args.sequence_length,
        device=args.device,
        atol=args.atol,
        rtol=args.rtol,
        expert_mode=args.expert_mode,
    )
    payload = report.to_dict()
    _write_json(args.output, payload)
    if args.html_output:
        write_eval_html_report_payload(report=payload, output_path=args.html_output)
    if args.print:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        status = "passed" if report.passed else "failed"
        print(f"{status}; wrote {args.output}")
    return 0 if report.passed or not args.strict else 1


def _cmd_eval_report_html(args: argparse.Namespace) -> int:
    write_eval_html_report(report_path=args.input, output_path=args.output)
    print(f"wrote {args.output}")
    return 0


def _cmd_eval_compare(args: argparse.Namespace) -> int:
    comparison = write_eval_comparison_report(
        report_paths=args.reports,
        output_path=args.output,
        html_output_path=args.html_output,
    )
    if args.print:
        print(json.dumps(comparison, indent=2, sort_keys=True))
    else:
        print(f"wrote {args.output}")
    return 0


def _cmd_eval_batch(args: argparse.Namespace) -> int:
    manifest = run_eval_batch(
        config_path=args.config,
        output_dir=args.output_dir,
        strict=True if args.strict else None,
    )
    if args.print:
        print(json.dumps(manifest, indent=2, sort_keys=True))
    else:
        print(f"wrote {Path(manifest['output_dir']) / 'eval-batch-manifest.json'}")
    has_errors = any(run.get("status") == "error" for run in manifest.get("runs", []))
    if has_errors:
        return 1
    if manifest.get("evaluation", {}).get("strict") and not manifest.get("passed"):
        return 1
    return 0


def _cmd_recovery_plan(args: argparse.Namespace) -> int:
    plan = write_recovery_plan(config_path=args.config, output_path=args.output)
    if args.print:
        print(json.dumps(plan, indent=2, sort_keys=True))
    else:
        print(f"wrote {plan['artifacts']['plan_path']}")
    return 0


def _cmd_recovery_run(args: argparse.Namespace) -> int:
    report = run_recovery(plan_path=args.plan, output_path=args.output, max_steps=args.max_steps)
    if args.print:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        output = args.output or Path(str(report["output_dir"])) / "recovery-run-report.json"
        print(f"wrote {output}")
    return 0


def _cmd_recovery_export(args: argparse.Namespace) -> int:
    report = export_recovered_wrapper(
        checkpoint_path=args.checkpoint,
        wrapper_dir=args.wrapper,
        output_dir=args.output_dir,
        artifact_name=args.artifact_name,
    )
    if args.print:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(f"wrote {report['output_dir']}")
    return 0


def _cmd_recovery_validate(args: argparse.Namespace) -> int:
    report = validate_recovered_wrapper(
        source_wrapper=args.source_wrapper,
        recovered_wrapper=args.recovered_wrapper,
        checkpoint_path=args.checkpoint,
        export_report_path=args.export_report,
        output_path=args.output,
    )
    if args.print:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(f"{report['status']}; wrote {args.output}")
    return 0 if report.get("passed") else 1


def _cmd_recovery_experiment(args: argparse.Namespace) -> int:
    report = run_recovery_experiment(
        config_path=args.config,
        output_dir=args.output_dir,
        max_steps=args.max_steps,
    )
    if args.print:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(f"wrote {report['artifacts']['json_report']}")
    return 0


def _cmd_recovery_compare(args: argparse.Namespace) -> int:
    comparison = write_recovery_comparison_report(
        report_paths=args.reports,
        output_path=args.output,
        html_output_path=args.html_output,
    )
    if args.print:
        print(json.dumps(comparison, indent=2, sort_keys=True))
    else:
        print(f"wrote {args.output}")
    return 0


def _cmd_model_card(args: argparse.Namespace) -> int:
    summary = write_model_card(
        wrapper_dir=args.wrapper,
        output_path=args.output,
        eval_reports=args.eval_report,
        recovery_reports=args.recovery_report,
        validation_reports=args.validation_report,
        commands=args.commands,
    )
    if args.print:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print(f"wrote {args.output}")
    return 0


def _cmd_smoke_assert(args: argparse.Namespace) -> int:
    output_path = args.output or args.run_dir / "smoke-assertions.json"
    report = assert_tiny_hf_smoke_run(run_dir=args.run_dir, output_path=output_path)
    if args.print:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(f"{report['status']}; wrote {output_path}")
    return 0 if report.get("passed") else 1


def _load_optional_texts(*, text: str | None, text_file: Path | None) -> list[str] | None:
    samples: list[str] = []
    if text:
        samples.append(text)
    if text_file:
        content = text_file.read_text(encoding="utf-8")
        samples.extend(chunk.strip() for chunk in content.split("\n\n") if chunk.strip())
    return samples or None


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
