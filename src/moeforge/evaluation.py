from __future__ import annotations

from dataclasses import asdict, dataclass, field
import time
from pathlib import Path
from typing import Any

from .hf_runtime import MoEForgeCarvedMLPModule, MoEForgeConfig, replace_hf_mlp_modules
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
    teacher_kl_loss: float | None
    dense_nll_loss: float | None
    carved_nll_loss: float | None
    nll_loss_delta: float | None
    loss_token_count: int
    active_experts: list[dict[str, Any]]


@dataclass(slots=True)
class EvalActiveExperts:
    sample_index: int
    layer: int
    experts: list[int]
    mode: str
    token_count: int | None = None
    top_k: int | None = None
    expert_token_counts: dict[str, int] | None = None
    mean_selected_weight_by_expert: dict[str, float] | None = None


@dataclass(slots=True)
class EvalLayerAttribution:
    sample_index: int
    layer: int
    experts: list[int]
    dense_vs_all_max_abs_error: float
    dense_vs_all_mean_abs_error: float
    dense_vs_selected_max_abs_error: float
    dense_vs_selected_mean_abs_error: float
    selected_vs_all_max_abs_error: float
    selected_vs_all_mean_abs_error: float


@dataclass(slots=True)
class EvalSummary:
    average_dense_latency_s: float
    average_carved_latency_s: float
    average_carved_vs_dense_latency_ratio: float | None
    worst_sample_index: int | None
    worst_sample_max_abs_error: float
    worst_layer_sample_index: int | None
    worst_layer: int | None
    worst_layer_selected_vs_all_max_abs_error: float
    average_teacher_kl_loss: float | None
    average_dense_nll_loss: float | None
    average_carved_nll_loss: float | None
    average_nll_loss_delta: float | None
    loss_token_count: int


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
    layer_attribution: list[EvalLayerAttribution]
    summary: EvalSummary
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
    replacement_payload = replacements.to_dict()
    attribution_modules = _load_attribution_modules(
        package_path=package_path,
        config=hf_config,
        replacements=replacement_payload,
        torch=torch,
    )
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
    layer_attribution: list[EvalLayerAttribution] = []
    with torch.no_grad():
        for index, prepared_sample in enumerate(prepared):
            sample_experts = _active_experts_for_sample(
                config=hf_config,
                router_plan=router_plan,
                sample=prepared_sample,
                mode=expert_mode,
            )
            captures = _capture_dense_mlp_io(dense, replacements=replacement_payload)
            try:
                dense_output, dense_latency = _timed_forward(dense, prepared_sample["inputs"], torch=torch)
            finally:
                captures["close"]()
            _apply_default_experts(carved, replacements=replacement_payload, experts_by_layer=sample_experts)
            carved_output, carved_latency = _timed_forward(carved, prepared_sample["inputs"], torch=torch)
            active_records = _active_records_for_sample(
                sample_index=index,
                model=carved,
                replacements=replacement_payload,
                experts_by_layer=sample_experts,
                mode=expert_mode,
            )
            diff = (dense_output.logits - carved_output.logits).abs()
            allclose = bool(torch.allclose(dense_output.logits, carved_output.logits, atol=atol, rtol=rtol))
            quality = _quality_metrics(
                dense_logits=dense_output.logits,
                carved_logits=carved_output.logits,
                inputs=prepared_sample["inputs"],
                torch=torch,
            )
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
                    teacher_kl_loss=quality["teacher_kl_loss"],
                    dense_nll_loss=quality["dense_nll_loss"],
                    carved_nll_loss=quality["carved_nll_loss"],
                    nll_loss_delta=quality["nll_loss_delta"],
                    loss_token_count=quality["loss_token_count"],
                    active_experts=[asdict(record) for record in active_records],
                )
            )
            layer_attribution.extend(
                _layer_attribution_for_sample(
                    sample_index=index,
                    captures=captures["records"],
                    modules=attribution_modules,
                    experts_by_layer=sample_experts,
                    mode=expert_mode,
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
            token_count=int(record["token_count"]) if record.get("token_count") is not None else None,
            top_k=int(record["top_k"]) if record.get("top_k") is not None else None,
            expert_token_counts=(
                {str(key): int(value) for key, value in record["expert_token_counts"].items()}
                if isinstance(record.get("expert_token_counts"), dict)
                else None
            ),
            mean_selected_weight_by_expert=(
                {str(key): float(value) for key, value in record["mean_selected_weight_by_expert"].items()}
                if isinstance(record.get("mean_selected_weight_by_expert"), dict)
                else None
            ),
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
        replacements=replacement_payload,
        active_experts=active_expert_records,
        layer_attribution=layer_attribution,
        summary=_summary(samples=samples, layer_attribution=layer_attribution),
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


def _load_attribution_modules(
    *,
    package_path: Path,
    config: MoEForgeConfig,
    replacements: dict[str, Any],
    torch: Any,
) -> dict[int, Any]:
    modules = {}
    for item in replacements.get("replaced", []):
        layer = int(item["layer"])
        module = MoEForgeCarvedMLPModule.from_package(package_path, layer=layer, config=config)
        dtype = _dtype_from_string(item.get("dtype"), torch=torch)
        device = torch.device(str(item.get("device") or "cpu"))
        if dtype is None:
            module = module.to(device=device)
        else:
            module = module.to(device=device, dtype=dtype)
        modules[layer] = module
    return modules


def _capture_dense_mlp_io(model: Any, *, replacements: dict[str, Any]) -> dict[str, Any]:
    records: dict[int, dict[str, Any]] = {}
    handles = []

    def make_hook(layer: int):
        def hook(_module: Any, inputs: tuple[Any, ...], output: Any) -> None:
            input_tensor = inputs[0] if inputs else None
            output_tensor = output[0] if isinstance(output, tuple) else output
            records[layer] = {
                "input": input_tensor.detach(),
                "output": output_tensor.detach(),
            }

        return hook

    for item in replacements.get("replaced", []):
        layer = int(item["layer"])
        module = model.get_submodule(str(item["module_path"]))
        handles.append(module.register_forward_hook(make_hook(layer)))

    def close() -> None:
        for handle in handles:
            handle.remove()

    return {"records": records, "close": close}


def _layer_attribution_for_sample(
    *,
    sample_index: int,
    captures: dict[int, dict[str, Any]],
    modules: dict[int, Any],
    experts_by_layer: dict[int, list[int]],
    mode: str,
) -> list[EvalLayerAttribution]:
    records = []
    for layer in sorted(modules):
        captured = captures.get(layer)
        if not captured:
            continue
        module = modules[layer]
        buffer = next(module.buffers())
        hidden = captured["input"].to(device=buffer.device, dtype=buffer.dtype)
        dense_output = captured["output"].to(device=hidden.device, dtype=hidden.dtype)
        experts = experts_by_layer.get(layer, [])
        all_output = module.forward_all(hidden)
        if mode == "learned-router":
            selected_output = module.forward_token_router(hidden)
            summary = module.last_router_summary or {}
            experts = [int(expert) for expert in summary.get("experts", [])]
        else:
            selected_output = module.forward_selected(hidden, experts=experts)
        dense_all = _tensor_diff(dense_output, all_output)
        dense_selected = _tensor_diff(dense_output, selected_output)
        selected_all = _tensor_diff(selected_output, all_output)
        records.append(
            EvalLayerAttribution(
                sample_index=sample_index,
                layer=layer,
                experts=experts,
                dense_vs_all_max_abs_error=dense_all["max_abs_error"],
                dense_vs_all_mean_abs_error=dense_all["mean_abs_error"],
                dense_vs_selected_max_abs_error=dense_selected["max_abs_error"],
                dense_vs_selected_mean_abs_error=dense_selected["mean_abs_error"],
                selected_vs_all_max_abs_error=selected_all["max_abs_error"],
                selected_vs_all_mean_abs_error=selected_all["mean_abs_error"],
            )
        )
    return records


def _summary(*, samples: list[EvalSample], layer_attribution: list[EvalLayerAttribution]) -> EvalSummary:
    average_dense = _average([sample.dense_latency_s for sample in samples])
    average_carved = _average([sample.carved_latency_s for sample in samples])
    loss_token_count = int(sum(sample.loss_token_count for sample in samples))
    worst_sample = max(samples, key=lambda sample: sample.max_abs_error, default=None)
    worst_layer = max(
        layer_attribution,
        key=lambda record: record.selected_vs_all_max_abs_error,
        default=None,
    )
    return EvalSummary(
        average_dense_latency_s=average_dense,
        average_carved_latency_s=average_carved,
        average_carved_vs_dense_latency_ratio=_latency_ratio(average_dense, average_carved),
        worst_sample_index=worst_sample.index if worst_sample else None,
        worst_sample_max_abs_error=worst_sample.max_abs_error if worst_sample else 0.0,
        worst_layer_sample_index=worst_layer.sample_index if worst_layer else None,
        worst_layer=worst_layer.layer if worst_layer else None,
        worst_layer_selected_vs_all_max_abs_error=(
            worst_layer.selected_vs_all_max_abs_error if worst_layer else 0.0
        ),
        average_teacher_kl_loss=_weighted_average_optional(
            [(sample.teacher_kl_loss, sample.loss_token_count) for sample in samples]
        ),
        average_dense_nll_loss=_weighted_average_optional(
            [(sample.dense_nll_loss, sample.loss_token_count) for sample in samples]
        ),
        average_carved_nll_loss=_weighted_average_optional(
            [(sample.carved_nll_loss, sample.loss_token_count) for sample in samples]
        ),
        average_nll_loss_delta=_weighted_average_optional(
            [(sample.nll_loss_delta, sample.loss_token_count) for sample in samples]
        ),
        loss_token_count=loss_token_count,
    )


def _average(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _weighted_average_optional(values: list[tuple[float | None, int]]) -> float | None:
    scored = [(float(value), int(weight)) for value, weight in values if value is not None and int(weight) > 0]
    total_weight = sum(weight for _value, weight in scored)
    if total_weight <= 0:
        return None
    return float(sum(value * weight for value, weight in scored) / total_weight)


def _tensor_diff(left: Any, right: Any) -> dict[str, float]:
    diff = (left - right).abs()
    return {
        "max_abs_error": float(diff.max().item()) if diff.numel() else 0.0,
        "mean_abs_error": float(diff.mean().item()) if diff.numel() else 0.0,
    }


def _quality_metrics(*, dense_logits: Any, carved_logits: Any, inputs: dict[str, Any], torch: Any) -> dict[str, Any]:
    input_ids = inputs.get("input_ids")
    if input_ids is None or input_ids.shape[-1] < 2:
        return {
            "teacher_kl_loss": None,
            "dense_nll_loss": None,
            "carved_nll_loss": None,
            "nll_loss_delta": None,
            "loss_token_count": 0,
        }
    labels = input_ids[:, 1:].contiguous()
    dense_shifted = dense_logits[:, :-1, :].float().contiguous()
    carved_shifted = carved_logits[:, :-1, :].float().contiguous()
    mask = _loss_mask(inputs=inputs, labels=labels, torch=torch)
    token_count = int(mask.sum().item())
    if token_count <= 0:
        return {
            "teacher_kl_loss": None,
            "dense_nll_loss": None,
            "carved_nll_loss": None,
            "nll_loss_delta": None,
            "loss_token_count": 0,
        }
    teacher_kl = torch.nn.functional.kl_div(
        torch.nn.functional.log_softmax(carved_shifted, dim=-1),
        torch.nn.functional.softmax(dense_shifted, dim=-1),
        reduction="none",
    ).sum(dim=-1)
    dense_nll = _token_nll(dense_shifted, labels=labels, torch=torch)
    carved_nll = _token_nll(carved_shifted, labels=labels, torch=torch)
    teacher_kl_loss = max(0.0, _masked_mean(teacher_kl, mask=mask))
    dense_nll_loss = _masked_mean(dense_nll, mask=mask)
    carved_nll_loss = _masked_mean(carved_nll, mask=mask)
    return {
        "teacher_kl_loss": teacher_kl_loss,
        "dense_nll_loss": dense_nll_loss,
        "carved_nll_loss": carved_nll_loss,
        "nll_loss_delta": carved_nll_loss - dense_nll_loss,
        "loss_token_count": token_count,
    }


def _loss_mask(*, inputs: dict[str, Any], labels: Any, torch: Any) -> Any:
    attention_mask = inputs.get("attention_mask")
    if attention_mask is None:
        return torch.ones_like(labels, dtype=torch.float32)
    return attention_mask[:, 1:].to(dtype=torch.float32)


def _token_nll(logits: Any, *, labels: Any, torch: Any) -> Any:
    vocab_size = logits.shape[-1]
    losses = torch.nn.functional.cross_entropy(
        logits.reshape(-1, vocab_size),
        labels.reshape(-1),
        reduction="none",
    )
    return losses.reshape(labels.shape)


def _masked_mean(values: Any, *, mask: Any) -> float:
    return float(((values * mask).sum() / mask.sum()).detach().cpu().item())


def _dtype_from_string(value: Any, *, torch: Any) -> Any | None:
    if value is None:
        return None
    name = str(value).replace("torch.", "")
    return getattr(torch, name, None)


def _normalize_expert_mode(expert_mode: str) -> str:
    if expert_mode not in {"all", "default-pool", "router", "learned-router"}:
        raise EvaluationError("expert_mode must be one of: all, default-pool, router, learned-router")
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
    elif mode == "learned-router":
        if config.token_router_top_k is None:
            raise EvaluationError("expert_mode=learned-router requires token_router_top_k in the wrapper package")
        return {}
    else:
        if router_plan is None:
            raise EvaluationError(f"expert_mode={mode} requires router metadata in the wrapper package")
        experts = select_expert_pool(
            router_plan,
            text=sample.get("text"),
            document_index=int(sample.get("document_index", 0)) if mode == "router" else None,
        )
    return {int(layer): list(experts) for layer in config.layer_ids()}


def _active_records_for_sample(
    *,
    sample_index: int,
    model: Any,
    replacements: dict[str, Any],
    experts_by_layer: dict[int, list[int]],
    mode: str,
) -> list[EvalActiveExperts]:
    records = []
    for item in replacements.get("replaced", []):
        layer = int(item["layer"])
        module = model.get_submodule(str(item["module_path"]))
        if mode == "learned-router":
            summary = module.last_router_summary
            if summary is None:
                raise EvaluationError(f"learned-router mode produced no router summary for layer {layer}")
            records.append(
                EvalActiveExperts(
                    sample_index=sample_index,
                    layer=layer,
                    experts=[int(expert) for expert in summary.get("experts", [])],
                    mode=mode,
                    token_count=int(summary.get("token_count", 0)),
                    top_k=int(summary.get("top_k", 0)) if summary.get("top_k") is not None else None,
                    expert_token_counts={
                        str(key): int(value)
                        for key, value in dict(summary.get("expert_token_counts", {})).items()
                    },
                    mean_selected_weight_by_expert={
                        str(key): float(value)
                        for key, value in dict(summary.get("mean_selected_weight_by_expert", {})).items()
                    },
                )
            )
        else:
            records.append(
                EvalActiveExperts(
                    sample_index=sample_index,
                    layer=layer,
                    experts=experts_by_layer.get(layer, []),
                    mode=mode,
                )
            )
    return records


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
