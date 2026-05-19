from __future__ import annotations

from dataclasses import asdict, dataclass, field
import time
from pathlib import Path
from typing import Any

from .hf_runtime import MoEForgeConfig, replace_hf_mlp_modules
from .router import select_expert_pool
from .wrapper import load_router_plan, load_wrapper_config


class EvaluationError(RuntimeError):
    """Raised when a dense-vs-carved evaluation cannot be run."""


@dataclass(slots=True)
class EvalSample:
    index: int
    source: str
    token_count: int
    max_abs_error: float
    mean_abs_error: float
    allclose: bool
    dense_latency_s: float
    carved_latency_s: float
    carved_vs_dense_latency_ratio: float | None
    expert_mode: str
    active_experts: list[dict[str, Any]]


@dataclass(slots=True)
class EvalActiveExperts:
    sample_index: int
    layer: int
    experts: list[int]
    mode: str


@dataclass(slots=True)
class EvalReport:
    model: str
    package_dir: str
    source_model: str
    adapter_family: str | None
    sample_count: int
    passed: bool
    max_abs_error: float
    mean_abs_error: float
    atol: float
    rtol: float
    replacements: dict[str, Any]
    active_experts: list[EvalActiveExperts]
    samples: list[EvalSample] = field(default_factory=list)
    memory: dict[str, Any] = field(default_factory=dict)
    package: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def evaluate_hf_dense_vs_carved(
    *,
    model: str | Path,
    package_dir: str | Path,
    texts: list[str] | None = None,
    input_ids: list[list[int]] | None = None,
    sequence_length: int = 128,
    device: str = "cpu",
    atol: float = 1e-5,
    rtol: float = 1e-5,
    expert_mode: str = "all",
) -> EvalReport:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:  # pragma: no cover - optional dependency boundary
        raise EvaluationError("HF evaluation requires torch and transformers") from exc

    package_path = Path(package_dir)
    wrapper_config_path = package_path / "moeforge_config.json"
    wrapper_config = load_wrapper_config(wrapper_config_path)
    hf_config = MoEForgeConfig.from_package(package_path)
    warnings = list(wrapper_config.warnings)
    target_device = _resolve_device(device, torch)
    router_plan = load_router_plan(wrapper_config_path) if wrapper_config.router_plan_path else None
    expert_mode = _normalize_expert_mode(expert_mode)

    dense = AutoModelForCausalLM.from_pretrained(str(model)).to(target_device)
    carved = AutoModelForCausalLM.from_pretrained(str(model)).to(target_device)
    dense.eval()
    carved.eval()
    replacements = replace_hf_mlp_modules(carved, package_path, config=hf_config)
    prepared = _prepare_inputs(
        model=model,
        dense=dense,
        texts=texts,
        input_ids=input_ids,
        sequence_length=sequence_length,
        device=target_device,
        tokenizer_cls=AutoTokenizer,
        torch=torch,
        warnings=warnings,
    )

    samples: list[EvalSample] = []
    with torch.no_grad():
        for index, prepared_sample in enumerate(prepared):
            sample_experts = _active_experts_for_sample(
                config=hf_config,
                router_plan=router_plan,
                sample=prepared_sample,
                mode=expert_mode,
            )
            dense_output, dense_latency = _timed_forward(dense, prepared_sample["inputs"], torch=torch)
            _apply_default_experts(carved, replacements=replacements.to_dict(), experts_by_layer=sample_experts)
            carved_output, carved_latency = _timed_forward(carved, prepared_sample["inputs"], torch=torch)
            diff = (dense_output.logits - carved_output.logits).abs()
            allclose = bool(torch.allclose(dense_output.logits, carved_output.logits, atol=atol, rtol=rtol))
            active_records = [
                EvalActiveExperts(
                    sample_index=index,
                    layer=layer,
                    experts=experts,
                    mode=expert_mode,
                )
                for layer, experts in sample_experts.items()
            ]
            samples.append(
                EvalSample(
                    index=index,
                    source=str(prepared_sample["source"]),
                    token_count=int(prepared_sample["token_count"]),
                    max_abs_error=float(diff.max().item()) if diff.numel() else 0.0,
                    mean_abs_error=float(diff.mean().item()) if diff.numel() else 0.0,
                    allclose=allclose,
                    dense_latency_s=dense_latency,
                    carved_latency_s=carved_latency,
                    carved_vs_dense_latency_ratio=_latency_ratio(dense_latency, carved_latency),
                    expert_mode=expert_mode,
                    active_experts=[asdict(record) for record in active_records],
                )
            )

    max_abs = max((sample.max_abs_error for sample in samples), default=0.0)
    mean_abs = float(sum(sample.mean_abs_error for sample in samples) / len(samples)) if samples else 0.0
    active_expert_records = [
        EvalActiveExperts(
            sample_index=int(sample.index),
            layer=int(record["layer"]),
            experts=[int(expert) for expert in record["experts"]],
            mode=str(record["mode"]),
        )
        for sample in samples
        for record in sample.active_experts
    ]
    return EvalReport(
        model=str(model),
        package_dir=str(package_path),
        source_model=hf_config.source_model,
        adapter_family=hf_config.adapter_family,
        sample_count=len(samples),
        passed=all(sample.allclose for sample in samples),
        max_abs_error=max_abs,
        mean_abs_error=mean_abs,
        atol=atol,
        rtol=rtol,
        replacements=replacements.to_dict(),
        active_experts=active_expert_records,
        samples=samples,
        memory=_memory_report(dense=dense, carved=carved, torch=torch, device=target_device),
        package=wrapper_config.to_dict(),
        warnings=warnings,
    )


def _prepare_inputs(
    *,
    model: str | Path,
    dense: Any,
    texts: list[str] | None,
    input_ids: list[list[int]] | None,
    sequence_length: int,
    device: Any,
    tokenizer_cls: Any,
    torch: Any,
    warnings: list[str],
) -> list[dict[str, Any]]:
    if input_ids is not None:
        normalized_input_ids = _normalize_input_ids(input_ids)
        return [
            {
                "source": f"input_ids:{index}",
                "document_index": index,
                "text": None,
                "token_count": len(sample),
                "inputs": {"input_ids": torch.tensor([sample], dtype=torch.long, device=device)},
            }
            for index, sample in enumerate(normalized_input_ids)
        ]

    if texts:
        try:
            tokenizer = tokenizer_cls.from_pretrained(str(model))
        except Exception as exc:  # pragma: no cover - depends on optional model assets
            raise EvaluationError("text evaluation requires a loadable tokenizer") from exc
        prepared = []
        for index, text in enumerate(texts):
            encoded = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=sequence_length,
            )
            prepared.append(
                {
                    "source": f"text:{index}",
                    "document_index": index,
                    "text": text,
                    "token_count": int(encoded["input_ids"].shape[-1]),
                    "inputs": {key: value.to(device) for key, value in encoded.items()},
                }
            )
        return prepared

    vocab_size = int(getattr(dense.config, "vocab_size", 32) or 32)
    length = max(1, min(sequence_length, vocab_size, 8))
    sample = [index % vocab_size for index in range(1, length + 1)]
    warnings.append("no evaluation samples were provided; used deterministic smoke-test token ids")
    return [
        {
            "source": "generated:smoke_input_ids",
            "document_index": 0,
            "text": None,
            "token_count": len(sample),
            "inputs": {"input_ids": torch.tensor([sample], dtype=torch.long, device=device)},
        }
    ]


def _timed_forward(model: Any, inputs: dict[str, Any], *, torch: Any) -> tuple[Any, float]:
    _synchronize_if_cuda(torch)
    start = time.perf_counter()
    output = model(**inputs)
    _synchronize_if_cuda(torch)
    return output, time.perf_counter() - start


def _normalize_expert_mode(expert_mode: str) -> str:
    if expert_mode not in {"all", "default-pool", "router"}:
        raise EvaluationError("expert_mode must be one of: all, default-pool, router")
    return expert_mode


def _active_experts_for_sample(
    *,
    config: MoEForgeConfig,
    router_plan: dict[str, Any] | None,
    sample: dict[str, Any],
    mode: str,
) -> dict[int, list[int]]:
    if mode == "all":
        experts = list(range(config.expert_count))
    else:
        if router_plan is None:
            raise EvaluationError(f"expert_mode={mode} requires router metadata in the wrapper package")
        experts = select_expert_pool(
            router_plan,
            text=sample.get("text"),
            document_index=int(sample.get("document_index", 0)) if mode == "router" else None,
        )
    return {int(layer): list(experts) for layer in config.layer_ids()}


def _apply_default_experts(
    model: Any,
    *,
    replacements: dict[str, Any],
    experts_by_layer: dict[int, list[int]],
) -> None:
    for item in replacements.get("replaced", []):
        layer = int(item["layer"])
        module = model.get_submodule(str(item["module_path"]))
        module.set_default_experts(experts_by_layer.get(layer))


def _latency_ratio(dense_latency: float, carved_latency: float) -> float | None:
    if dense_latency <= 0:
        return None
    return carved_latency / dense_latency


def _normalize_input_ids(input_ids: list[list[int]]) -> list[list[int]]:
    if not isinstance(input_ids, list) or not input_ids:
        raise EvaluationError("input_ids must be a non-empty JSON list of token id lists")
    normalized = []
    for sample in input_ids:
        if not isinstance(sample, list) or not sample:
            raise EvaluationError("each input_ids sample must be a non-empty list")
        try:
            normalized.append([int(item) for item in sample])
        except (TypeError, ValueError) as exc:
            raise EvaluationError("input_ids samples must contain integer token ids") from exc
    return normalized


def _synchronize_if_cuda(torch: Any) -> None:
    if torch.cuda.is_available():  # pragma: no cover - depends on local hardware
        torch.cuda.synchronize()


def _resolve_device(device: str, torch: Any) -> Any:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _memory_report(*, dense: Any, carved: Any, torch: Any, device: Any) -> dict[str, Any]:
    report: dict[str, Any] = {
        "device": str(device),
        "dense_parameter_count": _parameter_count(dense),
        "carved_parameter_count": _parameter_count(carved),
        "carved_buffer_count": _buffer_count(carved),
    }
    if str(device).startswith("cuda") and torch.cuda.is_available():  # pragma: no cover - depends on local hardware
        report["cuda_max_memory_allocated_bytes"] = int(torch.cuda.max_memory_allocated(device))
    else:
        report["cuda_max_memory_allocated_bytes"] = None
        report["note"] = "CUDA memory counters are unavailable for this run."
    return report


def _parameter_count(model: Any) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters()))


def _buffer_count(model: Any) -> int:
    return int(sum(buffer.numel() for buffer in model.buffers()))
