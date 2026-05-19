from __future__ import annotations

from html import escape
import json
from pathlib import Path
from typing import Any


class RecoveryCompareError(RuntimeError):
    """Raised when recovery experiment reports cannot be compared."""


def build_recovery_comparison(*, report_paths: list[Path]) -> dict[str, Any]:
    if len(report_paths) < 2:
        raise RecoveryCompareError("recovery comparison requires at least two reports")
    reports = []
    for index, path in enumerate(report_paths):
        payload = _load_report(path)
        reports.append(_comparison_record(payload, path=path, index=index))
    ranked = sorted(
        reports,
        key=lambda item: (
            item.get("validation_status") != "validated",
            _rank_number(item.get("average_teacher_kl_delta")),
            _rank_number(item.get("average_nll_delta_delta")),
            _rank_number(item.get("final_loss")),
            _rank_number(item.get("average_after_latency_ratio")),
        ),
    )
    for rank, item in enumerate(ranked, start=1):
        item["rank"] = rank
    return {
        "format": "moeforge_recovery_experiment_comparison",
        "ranking_policy": (
            "validated runs first, then lower average_teacher_kl_delta, "
            "lower average_nll_delta_delta, lower final_loss, and lower after latency ratio"
        ),
        "report_count": len(reports),
        "reports": reports,
        "ranked": ranked,
        "best": ranked[0] if ranked else None,
        "fastest_after": _fastest_after(reports),
        "lowest_kl_delta": _lowest(reports, "average_teacher_kl_delta"),
        "lowest_nll_delta": _lowest(reports, "average_nll_delta_delta"),
        "warnings": _warnings(reports),
    }


def write_recovery_comparison_report(
    *,
    report_paths: list[Path],
    output_path: Path,
    html_output_path: Path | None = None,
) -> dict[str, Any]:
    comparison = build_recovery_comparison(report_paths=report_paths)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(comparison, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if html_output_path is not None:
        html_output_path.parent.mkdir(parents=True, exist_ok=True)
        html_output_path.write_text(render_recovery_comparison_html(comparison), encoding="utf-8")
    return comparison


def render_recovery_comparison_html(comparison: dict[str, Any]) -> str:
    if not isinstance(comparison, dict):
        raise RecoveryCompareError("recovery comparison must be a JSON object")
    title = f"MoE Forge Recovery Comparison ({comparison.get('report_count', 0)} reports)"
    return "\n".join(
        [
            "<!doctype html>",
            '<html lang="en">',
            "<head>",
            '<meta charset="utf-8">',
            '<meta name="viewport" content="width=device-width, initial-scale=1">',
            f"<title>{escape(title)}</title>",
            f"<style>{_css()}</style>",
            "</head>",
            "<body>",
            '<main class="page">',
            '<header class="header">',
            "<div>",
            '<p class="eyebrow">MoE Forge recovery comparison</p>',
            f"<h1>{escape(title)}</h1>",
            '<p class="subtle">Before/after recovery quality and speed across experiment artifacts.</p>',
            "</div>",
            "</header>",
            _summary_cards(comparison),
            _warnings_section(comparison),
            _ranking_table(comparison),
            _mode_table(comparison),
            "</main>",
            "</body>",
            "</html>",
            "",
        ]
    )


def _load_report(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("format") != "moeforge_recovery_experiment":
        raise RecoveryCompareError(f"{path} must contain a moeforge_recovery_experiment report")
    return payload


def _comparison_record(payload: dict[str, Any], *, path: Path, index: int) -> dict[str, Any]:
    summary = _dict(payload.get("summary"))
    before_batch = _dict(payload.get("before_eval_batch"))
    after_batch = _dict(payload.get("after_eval_batch"))
    recovery_run = _dict(payload.get("recovery_run"))
    train_source = _dict(recovery_run.get("train_sample_source"))
    quality = _dict(_dict(payload.get("quality_trends")).get("before_after_quality"))
    before_after = _dict(payload.get("before_after_eval"))
    mode_source = _list(before_after.get("mode_deltas")) or _list(quality.get("modes"))
    modes = [_mode_record(item) for item in mode_source if isinstance(item, dict)]
    label = path.parent.name if path.name == "recovery-experiment-report.json" else path.stem
    return {
        "index": index,
        "path": str(path),
        "label": label,
        "model": payload.get("model"),
        "source_wrapper": payload.get("source_wrapper"),
        "recovered_wrapper": payload.get("recovered_wrapper"),
        "validation_status": summary.get("recovered_wrapper_validation_status"),
        "steps_completed": summary.get("steps_completed"),
        "initial_loss": _optional_float(summary.get("initial_loss")),
        "final_loss": _optional_float(summary.get("final_loss")),
        "total_loss_delta": _optional_float(summary.get("total_loss_delta")),
        "average_teacher_kl_delta": _optional_float(summary.get("average_teacher_kl_delta")),
        "average_nll_delta_delta": _optional_float(summary.get("average_nll_delta_delta")),
        "average_after_latency_ratio": _average(item.get("latency_ratio_after") for item in modes),
        "average_latency_ratio_delta": _average(item.get("latency_ratio_delta") for item in modes),
        "improved_modes_by_max_abs_error": summary.get("improved_modes_by_max_abs_error"),
        "regressed_modes_by_max_abs_error": summary.get("regressed_modes_by_max_abs_error"),
        "improved_modes_by_teacher_kl": summary.get("improved_modes_by_teacher_kl"),
        "regressed_modes_by_teacher_kl": summary.get("regressed_modes_by_teacher_kl"),
        "before_sample_source": before_batch.get("sample_source"),
        "after_sample_source": after_batch.get("sample_source"),
        "train_sample_source": train_source,
        "train_sample_kind": train_source.get("kind"),
        "train_sample_count": train_source.get("sample_count"),
        "train_token_counts": train_source.get("token_counts"),
        "modes": modes,
        "artifacts": payload.get("artifacts"),
    }


def _mode_record(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "expert_mode": item.get("expert_mode"),
        "max_abs_error_delta": _optional_float(item.get("max_abs_error_delta")),
        "teacher_kl_loss_before": _optional_float(item.get("teacher_kl_loss_before")),
        "teacher_kl_loss_after": _optional_float(item.get("teacher_kl_loss_after")),
        "teacher_kl_loss_delta": _optional_float(item.get("teacher_kl_loss_delta")),
        "nll_loss_delta_before": _optional_float(item.get("nll_loss_delta_before")),
        "nll_loss_delta_after": _optional_float(item.get("nll_loss_delta_after")),
        "nll_loss_delta_delta": _optional_float(item.get("nll_loss_delta_delta")),
        "latency_ratio_before": _optional_float(item.get("latency_ratio_before")),
        "latency_ratio_after": _optional_float(item.get("latency_ratio_after")),
        "latency_ratio_delta": _optional_float(item.get("latency_ratio_delta")),
        "loss_token_count_after": item.get("loss_token_count_after"),
    }


def _summary_cards(comparison: dict[str, Any]) -> str:
    best = _dict(comparison.get("best"))
    fastest = _dict(comparison.get("fastest_after"))
    lowest_kl = _dict(comparison.get("lowest_kl_delta"))
    lowest_nll = _dict(comparison.get("lowest_nll_delta"))
    cards = [
        _card("Best", best.get("label")),
        _card("Best KL Delta", lowest_kl.get("label")),
        _card("Best NLL Delta", lowest_nll.get("label")),
        _card("Fastest After", fastest.get("label")),
        _card("Reports", comparison.get("report_count")),
    ]
    return '<section class="cards">' + "".join(cards) + "</section>"


def _warnings_section(comparison: dict[str, Any]) -> str:
    warnings = _list(comparison.get("warnings"))
    if not warnings:
        return ""
    items = "".join(f"<li>{escape(_text(item))}</li>" for item in warnings)
    return f'<section class="warnings"><h2>Warnings</h2><ul>{items}</ul></section>'


def _ranking_table(comparison: dict[str, Any]) -> str:
    rows = []
    for item in _list(comparison.get("ranked")):
        if not isinstance(item, dict):
            continue
        rows.append(
            "<tr>"
            f"<td>{escape(_text(item.get('rank')))}</td>"
            f"<td>{escape(_text(item.get('label')))}</td>"
            f"<td>{escape(_text(item.get('train_sample_kind')))}</td>"
            f"<td>{escape(_text(item.get('validation_status')))}</td>"
            f"<td>{escape(_number(item.get('steps_completed')))}</td>"
            f"<td>{escape(_number(item.get('average_teacher_kl_delta')))}</td>"
            f"<td>{escape(_number(item.get('average_nll_delta_delta')))}</td>"
            f"<td>{escape(_number(item.get('average_after_latency_ratio')))}</td>"
            f"<td>{escape(_number(item.get('total_loss_delta')))}</td>"
            "</tr>"
        )
    return (
        "<section><h2>Ranked Experiments</h2>"
        '<div class="table-wrap"><table>'
        "<thead><tr><th>Rank</th><th>Run</th><th>Train Data</th><th>Validation</th>"
        "<th>Steps</th><th>Avg KL Delta</th><th>Avg NLL Delta</th><th>After Latency</th>"
        "<th>Loss Delta</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table></div></section>"
    )


def _mode_table(comparison: dict[str, Any]) -> str:
    rows = []
    for report in _list(comparison.get("ranked")):
        if not isinstance(report, dict):
            continue
        for mode in _list(report.get("modes")):
            if not isinstance(mode, dict):
                continue
            rows.append(
                "<tr>"
                f"<td>{escape(_text(report.get('label')))}</td>"
                f"<td>{escape(_text(mode.get('expert_mode')))}</td>"
                f"<td>{escape(_number(mode.get('teacher_kl_loss_delta')))}</td>"
                f"<td>{escape(_number(mode.get('nll_loss_delta_delta')))}</td>"
                f"<td>{escape(_number(mode.get('max_abs_error_delta')))}</td>"
                f"<td>{escape(_number(mode.get('latency_ratio_delta')))}</td>"
                f"<td>{escape(_text(mode.get('loss_token_count_after')))}</td>"
                "</tr>"
            )
    return (
        "<section><h2>Mode Deltas</h2>"
        '<div class="table-wrap"><table>'
        "<thead><tr><th>Run</th><th>Mode</th><th>KL Delta</th><th>NLL Delta Delta</th>"
        "<th>Max Error Delta</th><th>Latency Delta</th><th>Loss Tokens</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table></div></section>"
    )


def _warnings(reports: list[dict[str, Any]]) -> list[str]:
    warnings = []
    sample_kinds = sorted({str(item.get("train_sample_kind")) for item in reports if item.get("train_sample_kind")})
    if len(sample_kinds) > 1:
        warnings.append(f"Reports use different training sample kinds: {', '.join(sample_kinds)}")
    invalid = [item["label"] for item in reports if item.get("validation_status") != "validated"]
    if invalid:
        warnings.append(f"Some recovered wrappers did not validate: {', '.join(invalid)}")
    return warnings


def _fastest_after(reports: list[dict[str, Any]]) -> dict[str, Any] | None:
    scored = [item for item in reports if item.get("average_after_latency_ratio") is not None]
    if not scored:
        return None
    return min(scored, key=lambda item: float(item["average_after_latency_ratio"]))


def _lowest(reports: list[dict[str, Any]], key: str) -> dict[str, Any] | None:
    scored = [item for item in reports if item.get(key) is not None]
    if not scored:
        return None
    return min(scored, key=lambda item: float(item[key]))


def _rank_number(value: Any) -> tuple[int, float]:
    number = _optional_float(value)
    if number is None:
        return (1, float("inf"))
    return (0, number)


def _average(values: Any) -> float | None:
    numbers = [_optional_float(value) for value in values]
    numbers = [value for value in numbers if value is not None]
    if not numbers:
        return None
    return float(sum(numbers) / len(numbers))


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _card(label: str, value: Any) -> str:
    return f'<article class="card"><span>{escape(label)}</span><strong>{escape(_number(value))}</strong></article>'


def _number(value: Any) -> str:
    number = _optional_float(value)
    if number is None:
        return _text(value)
    if abs(number) >= 1000 or (number != 0 and abs(number) < 0.001):
        return f"{number:.4e}"
    return f"{number:.6g}"


def _text(value: Any) -> str:
    if value is None:
        return "-"
    return str(value)


def _css() -> str:
    return """
:root {
  color-scheme: light;
  --bg: #f7f4ee;
  --ink: #1f2528;
  --muted: #657074;
  --panel: #ffffff;
  --line: #d7d0c5;
  --accent: #146c5a;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  background: var(--bg);
  color: var(--ink);
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}
.page { max-width: 1180px; margin: 0 auto; padding: 32px 20px 48px; }
.header { display: flex; justify-content: space-between; gap: 24px; align-items: end; margin-bottom: 24px; }
.eyebrow { margin: 0 0 6px; color: var(--accent); font-size: 12px; text-transform: uppercase; letter-spacing: .08em; font-weight: 700; }
h1 { margin: 0; font-size: 32px; line-height: 1.15; }
h2 { margin: 30px 0 12px; font-size: 18px; }
.subtle { color: var(--muted); margin: 8px 0 0; }
.cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; }
.card { background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 14px; }
.card span { display: block; color: var(--muted); font-size: 12px; margin-bottom: 6px; }
.card strong { font-size: 20px; overflow-wrap: anywhere; }
.warnings { background: #fff7db; border: 1px solid #e5c767; border-radius: 8px; padding: 12px 16px; margin-top: 18px; }
.warnings h2 { margin-top: 0; }
.table-wrap { overflow-x: auto; background: var(--panel); border: 1px solid var(--line); border-radius: 8px; }
table { width: 100%; border-collapse: collapse; min-width: 900px; }
th, td { padding: 10px 12px; border-bottom: 1px solid var(--line); text-align: left; font-size: 13px; vertical-align: top; }
th { color: var(--muted); background: #fbfaf7; font-weight: 700; }
tr:last-child td { border-bottom: 0; }
""".strip()
