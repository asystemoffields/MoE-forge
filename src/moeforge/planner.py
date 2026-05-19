from __future__ import annotations

from dataclasses import dataclass

from .layout import _parse_layer_spec, build_layer_layouts, summarize_layouts
from .model_info import ModelInfo
from .recipe import CalibrationPlan, ConversionRecipe, EvalPlan, ExportPlan, RouterPlan, Stage


@dataclass(slots=True)
class PlanOptions:
    goal: str = "balanced"
    target: str = "hf"
    hardware: str = "auto"
    experts: int | None = None
    top_k: int | None = None
    shared_ratio: float | None = None
    moe_layers: str | None = None
    calibration_samples: int | None = None
    recover_steps: int | None = None


def plan_conversion(info: ModelInfo, options: PlanOptions) -> ConversionRecipe:
    experts = options.experts or _default_experts(options.goal, info)
    top_k = options.top_k or _default_top_k(options.goal, experts)
    shared_ratio = options.shared_ratio if options.shared_ratio is not None else _default_shared_ratio(options.goal)
    moe_layers = _parse_layers(options.moe_layers, info.layer_count, options.goal)
    warnings = list(info.warnings)
    strategy, strategy_warning = _choose_strategy(info, options)
    if strategy_warning:
        warnings.append(strategy_warning)
    if info.source_format == "gguf":
        warnings.append("GGUF input can be inspected and planned; weight surgery is scheduled through an extraction/export backend.")
    if info.source_format.startswith("hf"):
        checkpoint = info.metadata.get("checkpoint")
        if isinstance(checkpoint, dict) and checkpoint.get("has_weights") is False:
            warnings.append("HF config was found, but no local checkpoint weights were detected yet.")
    if info.dense is False:
        warnings.append("Input already reports experts; recipe will treat it as an MoE retargeting or analysis job.")
    if info.layer_count is None:
        warnings.append("Layer count is unknown; recipe uses symbolic layer selection.")
    if info.adapter_family is None:
        warnings.append("No architecture adapter matched this model yet; conversion may require a custom tensor map.")
    try:
        layout = summarize_layouts(
            build_layer_layouts(
                info,
                moe_layers,
                experts=experts,
                top_k=top_k,
                shared_ratio=shared_ratio,
            )
        )
    except ValueError as exc:
        layout = summarize_layouts([])
        warnings.append(f"Layer layout is pending because {exc}.")

    stages = [
        Stage("inspect", "Read model config, tensor metadata, architecture family, and dense/MoE status."),
        Stage("profile", "Collect FFN activation statistics on calibration text."),
        Stage("construct", f"Build {strategy} experts for selected MLP layers."),
        Stage("route", "Initialize router from activation clusters and shared-neuron frequency."),
        Stage("recover", "Run teacher-distilled recovery training within the configured budget."),
        Stage("evaluate", "Compare dense baseline and MoE candidate on quality, KL, speed, and memory."),
        Stage("export", f"Export artifact for {options.target}."),
    ]

    return ConversionRecipe(
        source_model=str(info.path),
        source_format=info.source_format,
        architecture=info.architecture,
        adapter_family=info.adapter_family,
        goal=options.goal,
        target=options.target,
        hardware=options.hardware,
        strategy=strategy,
        experts=experts,
        top_k=top_k,
        shared_ratio=shared_ratio,
        moe_layers=moe_layers,
        router=RouterPlan(
            kind="activation_cluster",
            load_balance_weight=0.01,
            entropy_weight=0.001,
            notes=[
                "Start with analytical routing from calibration activations.",
                "Back off to higher shared ratio if teacher KL rises during validation.",
            ],
        ),
        calibration=CalibrationPlan(
            samples=options.calibration_samples or _default_calibration_samples(options.goal),
            sequence_length=2048 if options.goal != "tiny" else 1024,
            dataset="auto",
        ),
        recovery_steps=options.recover_steps if options.recover_steps is not None else _default_recovery_steps(options.goal),
        eval=EvalPlan(
            perplexity=True,
            teacher_kl=True,
            smoke_prompts=True,
            speed=True,
            memory=True,
        ),
        export=ExportPlan(target=options.target, quantize_after_export=options.target == "gguf"),
        layout=layout,
        stages=stages,
        warnings=warnings,
        source_summary=info.to_dict(),
    )


def _choose_strategy(info: ModelInfo, options: PlanOptions) -> tuple[str, str | None]:
    if info.dense is False:
        return "moe_retarget", None
    if options.goal == "quality":
        preferred = "sparse_upcycle"
    elif options.goal == "tiny":
        preferred = "adapter_moe"
    else:
        preferred = "carved_mlp"

    supported = _supported_backends(info)
    if not supported or preferred in supported:
        return preferred, None

    for fallback in ("carved_mlp", "adapter_moe", "sparse_upcycle"):
        if fallback in supported:
            return (
                fallback,
                f"Requested goal prefers {preferred}, but adapter {info.adapter_family} currently supports {', '.join(supported)}; selected {fallback}.",
            )
    return preferred, None


def _supported_backends(info: ModelInfo) -> tuple[str, ...]:
    if not info.adapter:
        return ()
    backends = info.adapter.get("supported_backends")
    if isinstance(backends, list):
        return tuple(str(item) for item in backends)
    if isinstance(backends, tuple):
        return tuple(str(item) for item in backends)
    return ()


def _default_experts(goal: str, info: ModelInfo) -> int:
    if goal == "tiny":
        return 4
    if goal == "explore":
        return 16
    if info.layer_count and info.layer_count >= 32:
        return 8
    return 6


def _default_top_k(goal: str, experts: int) -> int:
    if goal == "speed":
        return 1
    if goal == "quality":
        return min(4, experts)
    return min(2, experts)


def _default_shared_ratio(goal: str) -> float:
    if goal == "speed":
        return 0.15
    if goal == "quality":
        return 0.35
    if goal == "tiny":
        return 0.25
    return 0.25


def _default_calibration_samples(goal: str) -> int:
    if goal == "tiny":
        return 128
    if goal == "quality":
        return 2048
    if goal == "explore":
        return 1024
    return 512


def _default_recovery_steps(goal: str) -> int:
    if goal == "speed":
        return 500
    if goal == "quality":
        return 3000
    if goal == "tiny":
        return 250
    return 1000


def _parse_layers(value: str | None, layer_count: int | None, goal: str) -> list[int] | str:
    if value:
        return _parse_layer_spec(value, layer_count)

    if layer_count is None:
        return "middle_to_final_layers"

    if goal == "quality":
        start = max(0, layer_count // 3)
    elif goal == "speed":
        start = max(0, layer_count // 4)
    else:
        start = max(0, layer_count // 4)
    return list(range(start, layer_count))
