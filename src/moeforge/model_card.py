from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .wrapper import load_wrapper_config


class ModelCardError(RuntimeError):
    """Raised when a model card cannot be generated."""


def write_model_card(
    *,
    wrapper_dir: Path,
    output_path: Path,
    eval_reports: list[Path] | None = None,
    recovery_reports: list[Path] | None = None,
    validation_reports: list[Path] | None = None,
    commands: list[str] | None = None,
) -> dict[str, Any]:
    card = build_model_card(
        wrapper_dir=wrapper_dir,
        eval_reports=eval_reports or [],
        recovery_reports=recovery_reports or [],
        validation_reports=validation_reports or [],
        commands=commands or [],
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(card, encoding="utf-8")
    return {
        "format": "moeforge_model_card",
        "wrapper_dir": str(wrapper_dir),
        "output_path": str(output_path),
        "eval_report_count": len(eval_reports or []),
        "recovery_report_count": len(recovery_reports or []),
        "validation_report_count": len(validation_reports or []),
        "command_count": len(commands or []),
    }


def build_model_card(
    *,
    wrapper_dir: Path,
    eval_reports: list[Path],
    recovery_reports: list[Path],
    validation_reports: list[Path],
    commands: list[str],
) -> str:
    config_path = wrapper_dir / "moeforge_config.json"
    if not config_path.exists():
        raise ModelCardError(f"wrapper config not found: {config_path}")
    config = load_wrapper_config(config_path)
    eval_payloads = [_load_json(path) for path in eval_reports]
    recovery_payloads = [_load_json(path) for path in recovery_reports]
    validation_payloads = [_load_json(path) for path in validation_reports]

    lines = [
        "# MoE Forge Model Card",
        "",
        "## Package",
        "",
        f"- Wrapper directory: `{wrapper_dir}`",
        f"- Source model: `{config.source_model}`",
        f"- Adapter family: `{config.adapter_family}`",
        f"- Expert count: `{config.expert_count}`",
        f"- Converted layers: `{_layer_span([layer.layer for layer in config.layers])}`",
        f"- Token router top-k: {_config_ref(config.token_router_top_k)}",
        f"- Carved artifact: `{config.artifact_path}`",
        f"- Learned router artifact: {_artifact_ref(config.token_router_path)}",
        f"- Router metadata: {_artifact_ref(config.router_plan_path)}",
        "",
        "## Loading",
        "",
        "```python",
        "import moeforge",
        "from transformers import AutoModelForCausalLM",
        "",
        f"model = AutoModelForCausalLM.from_pretrained({str(wrapper_dir)!r})",
        "```",
        "",
        "## Evidence",
        "",
    ]
    lines.extend(_eval_section(eval_reports, eval_payloads))
    lines.extend(_router_activity_section(eval_reports, eval_payloads))
    lines.extend(_recovery_section(recovery_reports, recovery_payloads))
    lines.extend(_validation_section(validation_reports, validation_payloads))
    lines.extend(_commands_section(commands))
    lines.extend(_warnings_section(config.warnings, eval_payloads, recovery_payloads, validation_payloads))
    lines.extend(
        [
            "## Intended Use",
            "",
            "Use this package as an inspectable MoE Forge candidate for dense-teacher comparison, router experiments, recovery training, and local native-HF smoke runs.",
            "",
            "## Artifacts",
            "",
            "- `moeforge_config.json`: package contract for MoE Forge runtimes",
            "- `config.json`: Transformers config registered by the `moeforge` Python package",
            "- `carve-manifest.json`: dense-to-expert tensor slicing manifest",
            f"- `{config.artifact_path}`: carved shared/expert FFN tensors",
        ]
    )
    if config.token_router_path:
        lines.append(f"- `{config.token_router_path}`: learned per-token router weights")
    if config.router_plan_path:
        lines.append(f"- `{config.router_plan_path}`: document-pool router metadata")
    lines.append("")
    return "\n".join(lines)


def _eval_section(paths: list[Path], reports: list[dict[str, Any]]) -> list[str]:
    if not reports:
        return ["No eval reports were attached.", ""]
    lines = [
        "### Evaluation Reports",
        "",
        "| Report | Mode | Passed | Max Abs | Teacher KL | NLL Delta | Latency Ratio | Active Experts |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for path, report in zip(paths, reports):
        summary = _dict(report.get("summary"))
        modes = _modes(report)
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{_display_path(path)}`",
                    ", ".join(modes),
                    str(bool(report.get("passed"))),
                    _number(report.get("max_abs_error")),
                    _number(summary.get("average_teacher_kl_loss")),
                    _number(summary.get("average_nll_loss_delta")),
                    _number(summary.get("average_carved_vs_dense_latency_ratio")),
                    _active_summary(report),
                ]
            )
            + " |"
        )
    lines.append("")
    return lines


def _router_activity_section(paths: list[Path], reports: list[dict[str, Any]]) -> list[str]:
    rows: list[str] = []
    for path, report in zip(paths, reports):
        for row in _router_activity_rows(report):
            rows.append(
                "| "
                + " | ".join(
                    [
                        f"`{_display_path(path)}`",
                        _text(row["layer"]),
                        _text(row["mode"]),
                        _text(row["top_k"]),
                        _text(row["token_count"]),
                        _format_expert_map(row["expert_token_counts"]),
                        _format_float_map(row["mean_selected_weight_by_expert"]),
                    ]
                )
                + " |"
            )
    if not rows:
        return []
    return [
        "### Router Activity",
        "",
        "| Report | Layer | Mode | Top K | Tokens | Expert Token Counts | Mean Selected Weights |",
        "| --- | ---: | --- | ---: | ---: | --- | --- |",
        *rows,
        "",
    ]


def _recovery_section(paths: list[Path], reports: list[dict[str, Any]]) -> list[str]:
    if not reports:
        return ["### Recovery", "", "No recovery reports were attached.", ""]
    lines = [
        "### Recovery",
        "",
        "| Report | Steps | Initial Loss | Final Loss | Updated Tensors | Validation |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for path, report in zip(paths, reports):
        summary = _dict(report.get("summary"))
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{_display_path(path)}`",
                    _text(report.get("steps_completed") or summary.get("steps_completed")),
                    _number(report.get("initial_loss") or summary.get("initial_loss")),
                    _number(report.get("final_loss") or summary.get("final_loss")),
                    _text(summary.get("recovered_updated_tensor_count") or report.get("updated_tensor_count")),
                    _text(summary.get("recovered_wrapper_validation_status") or report.get("status")),
                ]
            )
            + " |"
        )
    lines.append("")
    return lines


def _validation_section(paths: list[Path], reports: list[dict[str, Any]]) -> list[str]:
    if not reports:
        return ["### Validation", "", "No validation reports were attached.", ""]
    lines = [
        "### Validation",
        "",
        "| Report | Status | Errors | Reloaded Layers | Native Load | Router Tensors | Changed Tensors |",
        "| --- | --- | ---: | ---: | --- | --- | ---: |",
    ]
    for path, report in zip(paths, reports):
        reload = _dict(report.get("reload"))
        comparison = _dict(report.get("tensor_comparison"))
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{_display_path(path)}`",
                    _text(report.get("status")),
                    str(len(_list(report.get("errors")))),
                    _text(reload.get("loaded_layer_count")),
                    _native_load_summary(report),
                    _router_tensor_summary(report),
                    _text(comparison.get("changed_tensor_count")),
                ]
            )
            + " |"
        )
    lines.append("")
    return lines


def _native_load_summary(report: dict[str, Any]) -> str:
    native = _dict(report.get("native_load"))
    if not native:
        return ""
    pieces = [_text(native.get("status"))]
    replaced = native.get("replaced_layer_count")
    routers = native.get("token_router_layer_count")
    if replaced is not None:
        pieces.append(f"{_text(replaced)} layers")
    if routers is not None:
        pieces.append(f"{_text(routers)} routers")
    return " / ".join(piece for piece in pieces if piece)


def _router_tensor_summary(report: dict[str, Any]) -> str:
    router = _dict(report.get("router_tensor_validation"))
    if not router:
        return ""
    missing = len(_list(router.get("missing_expected")))
    return f"{_text(router.get('tensor_count'))}/{_text(router.get('expected_tensor_count'))}; missing {missing}"


def _commands_section(commands: list[str]) -> list[str]:
    if not commands:
        return ["## Reproduction Commands", "", "No commands were attached.", ""]
    lines = ["## Reproduction Commands", "", "```powershell"]
    lines.extend(commands)
    lines.extend(["```", ""])
    return lines


def _warnings_section(
    wrapper_warnings: list[str],
    eval_reports: list[dict[str, Any]],
    recovery_reports: list[dict[str, Any]],
    validation_reports: list[dict[str, Any]],
) -> list[str]:
    warnings = [str(item) for item in wrapper_warnings]
    for payload in [*eval_reports, *recovery_reports, *validation_reports]:
        warnings.extend(str(item) for item in _list(payload.get("warnings")))
        warnings.extend(str(item) for item in _list(payload.get("errors")))
    if not warnings:
        return ["## Warnings And Assumptions", "", "No warnings were recorded in attached artifacts.", ""]
    return [
        "## Warnings And Assumptions",
        "",
        *[f"- {warning}" for warning in dict.fromkeys(warnings)],
        "",
    ]


def _active_summary(report: dict[str, Any]) -> str:
    by_layer = {}
    for item in _list(report.get("active_experts")):
        if not isinstance(item, dict):
            continue
        layer = item.get("layer")
        experts = item.get("experts")
        if layer is None or not isinstance(experts, list):
            continue
        by_layer.setdefault(str(layer), set()).add(tuple(int(expert) for expert in experts))
    if not by_layer:
        return ""
    pieces = []
    for layer, sets in sorted(by_layer.items(), key=lambda item: int(item[0])):
        rendered = ",".join("[" + ",".join(str(expert) for expert in experts) + "]" for experts in sorted(sets))
        pieces.append(f"L{layer}:{rendered}")
    return "; ".join(pieces)


def _router_activity_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    aggregated: dict[tuple[str, str, str], dict[str, Any]] = {}
    for item in _list(report.get("active_experts")):
        if not isinstance(item, dict):
            continue
        layer = _text(item.get("layer"))
        if not layer:
            continue
        mode = _text(item.get("mode")) or "unknown"
        top_k = _text(item.get("top_k"))
        key = (layer, mode, top_k)
        token_count = _int(item.get("token_count"))
        row = aggregated.setdefault(
            key,
            {
                "layer": layer,
                "mode": mode,
                "top_k": top_k,
                "token_count": 0,
                "expert_token_counts": {},
                "mean_selected_weight_by_expert": {},
                "_weight_denominators": {},
            },
        )
        row["token_count"] += token_count
        for expert, count in _dict(item.get("expert_token_counts")).items():
            row["expert_token_counts"][str(expert)] = row["expert_token_counts"].get(str(expert), 0) + _int(count)
        weight_denominator = token_count or 1
        for expert, weight in _dict(item.get("mean_selected_weight_by_expert")).items():
            expert_key = str(expert)
            row["mean_selected_weight_by_expert"][expert_key] = row["mean_selected_weight_by_expert"].get(
                expert_key, 0.0
            ) + (_float(weight) * weight_denominator)
            row["_weight_denominators"][expert_key] = row["_weight_denominators"].get(expert_key, 0) + weight_denominator

    rows = []
    for row in aggregated.values():
        denominators = row.pop("_weight_denominators")
        row["mean_selected_weight_by_expert"] = {
            expert: total / denominators[expert]
            for expert, total in row["mean_selected_weight_by_expert"].items()
            if denominators.get(expert)
        }
        rows.append(row)
    return sorted(rows, key=lambda row: (_int(row["layer"]), row["mode"], row["top_k"]))


def _modes(report: dict[str, Any]) -> list[str]:
    modes = {
        str(sample.get("expert_mode"))
        for sample in _list(report.get("samples"))
        if isinstance(sample, dict) and sample.get("expert_mode") is not None
    }
    return sorted(modes) or ["unknown"]


def _layer_span(layers: list[int]) -> str:
    if not layers:
        return ""
    sorted_layers = sorted(layers)
    if sorted_layers == list(range(sorted_layers[0], sorted_layers[-1] + 1)):
        return f"{sorted_layers[0]}..{sorted_layers[-1]} ({len(sorted_layers)} layers)"
    return ", ".join(str(layer) for layer in sorted_layers)


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ModelCardError(f"{path} must contain a JSON object")
    return payload


def _display_path(path: Path) -> str:
    return path.name if path.is_absolute() else path.as_posix()


def _artifact_ref(value: str | None) -> str:
    return f"`{value}`" if value else "not attached"


def _config_ref(value: Any) -> str:
    return f"`{value}`" if value is not None and _text(value) else "not configured"


def _format_expert_map(value: dict[str, int]) -> str:
    return ", ".join(f"{expert}:{value[expert]}" for expert in _sorted_keys(value))


def _format_float_map(value: dict[str, float]) -> str:
    return ", ".join(f"{expert}:{value[expert]:.4g}" for expert in _sorted_keys(value))


def _sorted_keys(value: dict[str, Any]) -> list[str]:
    return sorted(value, key=lambda item: (_int(item), item))


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _text(value: Any) -> str:
    return "" if value is None else str(value)


def _int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _number(value: Any) -> str:
    if value is None:
        return ""
    try:
        return f"{float(value):.6g}"
    except (TypeError, ValueError):
        return str(value)
