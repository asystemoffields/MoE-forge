from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class RunSummaryError(RuntimeError):
    """Raised when a run summary cannot be produced."""


# Heuristic thresholds for turning raw metrics into a verdict.
ALL_EXPERT_LOSSLESS_KL = 0.01  # all-experts teacher-KL below this == carve still lossless.
ROUTING_GAP_SMALL_KL = 0.1     # learned-router teacher-KL below this == near-dense.
ROUTING_GAP_MODERATE_KL = 0.5  # below this == promising; above == large gap.


def summarize_run(*, report_path: Path, output_path: Path | None = None) -> dict[str, Any]:
    """Turn a recovery-experiment / modal-recovery-manifest report into a decision-ready verdict."""
    payload = _load_json(report_path)
    summary = _dict(payload.get("summary"))
    before_after = _dict(_dict(payload.get("quality_trends")).get("before_after_quality"))
    modes = {
        str(item.get("expert_mode")): item
        for item in _list(before_after.get("modes"))
        if isinstance(item, dict)
    }
    learned = _dict(modes.get("learned-router"))
    all_experts = _dict(modes.get("all"))
    recovery_run = _dict(payload.get("recovery_run"))

    has_eval = bool(modes)
    training = _training_metrics(summary=summary, recovery_run=recovery_run)
    learned_metrics = _mode_metrics(learned)
    all_metrics = _mode_metrics(all_experts)

    experts_trained = int(summary.get("recovered_updated_tensor_count") or 0) > 0
    lossless = (
        all_metrics["kl_after"] is not None and all_metrics["kl_after"] < ALL_EXPERT_LOSSLESS_KL
    )
    routing_gap = _routing_gap_band(learned_metrics["kl_after"])
    direction = _direction(learned_metrics)
    undertrained = bool(training["still_improving"]) and direction in {"improving", "mixed"}

    verdicts = {
        "has_before_after_eval": has_eval,
        "experts_trained": experts_trained,
        "carve_lossless": lossless,
        "routing_gap": routing_gap,
        "direction": direction,
        "undertrained": undertrained,
    }

    findings = _findings(
        training=training,
        learned=learned_metrics,
        all_experts=all_metrics,
        verdicts=verdicts,
    )
    headline = _headline(learned=learned_metrics, verdicts=verdicts)
    next_commands = _next_commands(
        report_path=report_path,
        verdicts=verdicts,
        recovery_run=recovery_run,
    )

    report = {
        "format": "moeforge_run_summary",
        "report": str(report_path),
        "status": "summarized" if has_eval else "limited",
        "headline": headline,
        "findings": findings,
        "verdicts": verdicts,
        "metrics": {
            "training": training,
            "learned_router": learned_metrics,
            "all_experts": all_metrics,
        },
        "next_commands": next_commands,
    }
    if output_path is not None:
        _write_json(output_path, report)
    return report


def _training_metrics(*, summary: dict[str, Any], recovery_run: dict[str, Any]) -> dict[str, Any]:
    loss_points = [
        _num(item.get("total_loss"))
        for item in _list(recovery_run.get("losses"))
        if isinstance(item, dict) and _num(item.get("total_loss")) is not None
    ]
    initial = _num(summary.get("initial_loss"))
    final = _num(summary.get("final_loss"))
    min_loss = min(loss_points) if loss_points else None
    # "Still improving" == the final loss is at (or within noise of) the lowest seen, i.e. the
    # curve had not turned back up by the last step, so more steps plausibly keep helping.
    still_improving = (
        final is not None and min_loss is not None and final <= min_loss + 1e-6
    )
    return {
        "initial_loss": initial,
        "final_loss": final,
        "min_loss": min_loss,
        "loss_point_count": len(loss_points),
        "steps_completed": summary.get("steps_completed"),
        "trainable_parameter_count": recovery_run.get("trainable_parameter_count"),
        "updated_expert_tensor_count": summary.get("recovered_updated_tensor_count"),
        "updated_router_tensor_count": summary.get("recovered_updated_router_tensor_count"),
        "still_improving": still_improving,
    }


def _mode_metrics(mode: dict[str, Any]) -> dict[str, Any]:
    return {
        "kl_before": _num(mode.get("teacher_kl_loss_before")),
        "kl_after": _num(mode.get("teacher_kl_loss_after")),
        "kl_delta": _num(mode.get("teacher_kl_loss_delta")),
        "nll_before": _num(mode.get("carved_nll_loss_before")),
        "nll_after": _num(mode.get("carved_nll_loss_after")),
        "nll_delta": _num(mode.get("carved_nll_loss_delta")),
    }


def _routing_gap_band(kl_after: float | None) -> str:
    if kl_after is None:
        return "unknown"
    if kl_after < ROUTING_GAP_SMALL_KL:
        return "small"
    if kl_after < ROUTING_GAP_MODERATE_KL:
        return "moderate"
    return "large"


def _direction(learned: dict[str, Any]) -> str:
    kl_delta = learned["kl_delta"]
    nll_delta = learned["nll_delta"]
    if kl_delta is None and nll_delta is None:
        return "unknown"
    kl_better = kl_delta is not None and kl_delta < 0
    nll_better = nll_delta is not None and nll_delta < 0
    kl_worse = kl_delta is not None and kl_delta > 0
    nll_worse = nll_delta is not None and nll_delta > 0
    if kl_better and nll_better:
        return "improving"
    if (kl_worse and nll_worse) or (kl_worse and nll_delta is None):
        return "regressing"
    if (kl_better and nll_worse) or (kl_worse and nll_better):
        return "mixed"
    if kl_better or nll_better:
        return "improving"
    return "flat"


def _headline(*, learned: dict[str, Any], verdicts: dict[str, Any]) -> str:
    kl_after = learned["kl_after"]
    if not verdicts["has_before_after_eval"]:
        return "No before/after eval found in this report; nothing to score."
    if verdicts["routing_gap"] == "small":
        return f"Sparse MoE closely matches the dense teacher (learned-router KL {_fmt(kl_after)}) - near dense quality."
    if verdicts["direction"] == "improving":
        if verdicts["undertrained"]:
            return (
                f"Recovery improving on both KL and NLL but undertrained (loss still falling); "
                f"sparse-routing gap still {verdicts['routing_gap']} (KL {_fmt(kl_after)}) - train longer."
            )
        return (
            f"Recovery improved KL and NLL; sparse-routing gap remains {verdicts['routing_gap']} "
            f"(KL {_fmt(kl_after)})."
        )
    if verdicts["direction"] == "mixed":
        return (
            f"Router fitting the teacher but task NLL not improving (KL {_fmt(kl_after)}) - "
            f"likely capacity-limited; consider training experts or higher top-k."
        )
    if verdicts["direction"] == "regressing":
        return f"Recovery regressed quality (learned-router KL {_fmt(kl_after)}) - check LR / config."
    return f"Recovery did not reduce the sparse-routing gap (KL {_fmt(kl_after)}) - plateau."


def _findings(
    *,
    training: dict[str, Any],
    learned: dict[str, Any],
    all_experts: dict[str, Any],
    verdicts: dict[str, Any],
) -> list[str]:
    findings: list[str] = []
    findings.append(
        f"Training: loss {_fmt(training['initial_loss'])} -> {_fmt(training['final_loss'])} "
        f"over {training.get('steps_completed')} steps"
        + ("; loss still falling at the end (likely undertrained)." if training["still_improving"] else ".")
    )
    if verdicts["experts_trained"]:
        findings.append(
            f"Experts were trained ({training.get('updated_expert_tensor_count')} expert tensors, "
            f"{training.get('trainable_parameter_count')} params) - joint expert+router recovery."
        )
    else:
        findings.append("Router-only recovery (experts frozen); expert capacity unchanged.")
    if not verdicts["has_before_after_eval"]:
        findings.append("No learned-router before/after eval modes present; metrics limited.")
        return findings
    findings.append(
        f"Learned-router teacher-KL {_fmt(learned['kl_before'])} -> {_fmt(learned['kl_after'])} "
        f"(delta {_fmt(learned['kl_delta'], signed=True)}); "
        f"carved NLL {_fmt(learned['nll_before'])} -> {_fmt(learned['nll_after'])} "
        f"(delta {_fmt(learned['nll_delta'], signed=True)})."
    )
    if all_experts["kl_after"] is not None:
        if verdicts["carve_lossless"]:
            findings.append(
                f"All-experts reconstruction still ~lossless (KL {_fmt(all_experts['kl_after'])})."
            )
        else:
            findings.append(
                f"All-experts reconstruction degraded to KL {_fmt(all_experts['kl_after'])} "
                f"(from {_fmt(all_experts['kl_before'])}) - training experts moved them off the exact dense carve."
            )
    findings.append(
        f"Sparse-routing gap is {verdicts['routing_gap']}; recovery direction: {verdicts['direction']}."
    )
    return findings


def _next_commands(
    *,
    report_path: Path,
    verdicts: dict[str, Any],
    recovery_run: dict[str, Any],
) -> list[str]:
    commands: list[str] = []
    if not verdicts["has_before_after_eval"]:
        commands.append("Run a recovery-experiment with before/after eval modes to score this run.")
        return commands
    if verdicts["undertrained"]:
        commands.append("Train longer (raise --steps); the loss curve had not flattened.")
    if verdicts["routing_gap"] == "large" and not verdicts["undertrained"]:
        commands.append(
            "Plateau at a large gap: try higher --token-router-top-k, or switch to the sparse-upcycle backend."
        )
    if verdicts["direction"] == "mixed" and not verdicts["experts_trained"]:
        commands.append("Add --train-experts so recovery can move expert capacity, not just the router.")
    commands.append(
        "Benchmark the recovered wrapper (benchmark-plan + benchmark-compare) for a retention number vs dense."
    )
    return commands


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise RunSummaryError(f"could not read report: {path}") from exc
    except json.JSONDecodeError as exc:
        raise RunSummaryError(f"report is not valid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise RunSummaryError(f"report must be a JSON object: {path}")
    return payload


def _num(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _fmt(value: float | None, *, signed: bool = False) -> str:
    if value is None:
        return "n/a"
    return f"{value:+.4f}" if signed else f"{value:.4f}"


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
