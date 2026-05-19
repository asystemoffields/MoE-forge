from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .adapters import ADAPTERS
from .batch import run_eval_batch
from .carve import build_carve_manifest
from .evaluation import evaluate_hf_dense_vs_carved
from .inspectors import inspect_model
from .materialize import materialize_carve_manifest
from .planner import PlanOptions, plan_conversion
from .profiling import ProfileOptions, load_calibration_texts, profile_hf_model
from .reports import (
    write_eval_comparison_report,
    write_eval_html_report,
    write_eval_html_report_payload,
)
from .recovery import write_recovery_plan
from .recipe import recipe_to_dict
from .router import build_router_plan
from .runtime import verify_carved_artifact
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
    plan_parser.add_argument("--moe-layers", help="Layer range or list, such as 8:34 or 8,10,12.")
    plan_parser.add_argument("--calibration-samples", type=int, help="Calibration text sample count.")
    plan_parser.add_argument("--recover-steps", type=int, help="Recovery training step count.")
    plan_parser.add_argument("--output", type=Path, default=Path("recipe.json"), help="Recipe output path.")
    plan_parser.add_argument("--print", action="store_true", help="Also print the recipe JSON.")

    profile_parser = subparsers.add_parser("profile", help="Profile FFN activations on calibration text.")
    profile_parser.add_argument("model", help="Path to a local HF model folder or HF model id.")
    profile_parser.add_argument("--text", help="Inline calibration text sample.")
    profile_parser.add_argument("--text-file", type=Path, help="Calibration text file; blank lines split samples.")
    profile_parser.add_argument("--layers", help="Layer range or list, such as 8:34 or 8,10,12.")
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
        choices=["all", "default-pool", "router"],
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

    return parser


def _cmd_inspect(args: argparse.Namespace) -> int:
    info = inspect_model(args.model)
    payload = info.to_dict()
    if args.output:
        _write_json(args.output, payload)
    if args.json or not args.output:
        print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


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
