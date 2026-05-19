from __future__ import annotations

from html import escape
import json
from pathlib import Path
from typing import Any


class ReportError(RuntimeError):
    """Raised when a report artifact cannot be rendered."""


def build_eval_comparison(*, report_paths: list[Path]) -> dict[str, Any]:
    if len(report_paths) < 2:
        raise ReportError("comparison requires at least two eval reports")
    reports = []
    for index, path in enumerate(report_paths):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ReportError(f"{path} must contain an eval report JSON object")
        reports.append(_comparison_record(payload, path=path, index=index))
    ranked = sorted(
        reports,
        key=lambda item: (
            not bool(item["passed"]),
            _rank_number(item["max_abs_error"]),
            _rank_number(item["latency_ratio"]),
        ),
    )
    for rank, item in enumerate(ranked, start=1):
        item["rank"] = rank
    fastest = min(reports, key=lambda item: _rank_number(item["latency_ratio"]))
    lowest_error = min(reports, key=lambda item: _rank_number(item["max_abs_error"]))
    return {
        "format": "moeforge_eval_comparison",
        "ranking_policy": "passed first, then max_abs_error, then latency_ratio",
        "report_count": len(reports),
        "reports": reports,
        "ranked": ranked,
        "best": ranked[0] if ranked else None,
        "fastest": fastest,
        "lowest_error": lowest_error,
        "warnings": _comparison_warnings(reports),
    }


def write_eval_comparison_report(
    *,
    report_paths: list[Path],
    output_path: Path,
    html_output_path: Path | None = None,
) -> dict[str, Any]:
    comparison = build_eval_comparison(report_paths=report_paths)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(comparison, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if html_output_path is not None:
        html_output_path.parent.mkdir(parents=True, exist_ok=True)
        html_output_path.write_text(render_eval_comparison_html(comparison), encoding="utf-8")
    return comparison


def write_eval_html_report(*, report_path: Path, output_path: Path) -> None:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_eval_html_report(report), encoding="utf-8")


def write_eval_html_report_payload(*, report: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_eval_html_report(report), encoding="utf-8")


def render_eval_html_report(report: dict[str, Any]) -> str:
    if not isinstance(report, dict):
        raise ReportError("eval report must be a JSON object")
    title = f"MoE Forge Eval Report: {_text(report.get('model', 'model'))}"
    status = "passed" if report.get("passed") else "review"
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
            _header(report, status=status),
            _summary_cards(report),
            _warnings(report),
            _samples_table(report),
            _active_experts(report),
            _layer_attribution(report),
            _metadata(report),
            "</main>",
            "</body>",
            "</html>",
            "",
        ]
    )


def render_eval_comparison_html(comparison: dict[str, Any]) -> str:
    if not isinstance(comparison, dict):
        raise ReportError("comparison report must be a JSON object")
    title = f"MoE Forge Eval Comparison ({comparison.get('report_count', 0)} reports)"
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
            f"""
<header class="header">
  <div>
    <p class="eyebrow">MoE Forge comparison</p>
    <h1>{escape(title)}</h1>
    <p class="subtle">Quality/speed ranking across eval JSON artifacts.</p>
  </div>
</header>
""",
            _comparison_summary(comparison),
            _comparison_warnings_section(comparison),
            _comparison_table(comparison),
            _comparison_experts(comparison),
            "</main>",
            "</body>",
            "</html>",
            "",
        ]
    )


def _header(report: dict[str, Any], *, status: str) -> str:
    return f"""
<header class="header">
  <div>
    <p class="eyebrow">MoE Forge evaluation</p>
    <h1>{escape(_text(report.get("model", "model")))}</h1>
    <p class="subtle">{escape(_text(report.get("package_dir", "")))}</p>
  </div>
  <div class="status {status}">{escape(status)}</div>
</header>
"""


def _comparison_record(payload: dict[str, Any], *, path: Path, index: int) -> dict[str, Any]:
    summary = _dict(payload.get("summary"))
    samples = _list(payload.get("samples"))
    modes = sorted(
        {
            str(sample.get("expert_mode"))
            for sample in samples
            if isinstance(sample, dict) and sample.get("expert_mode") is not None
        }
    )
    active_experts = _summarize_active_experts(payload)
    return {
        "index": index,
        "path": str(path),
        "label": path.stem,
        "model": payload.get("model"),
        "package_dir": payload.get("package_dir"),
        "passed": bool(payload.get("passed")),
        "sample_count": int(payload.get("sample_count") or 0),
        "expert_modes": modes,
        "max_abs_error": _optional_float(payload.get("max_abs_error")),
        "mean_abs_error": _optional_float(payload.get("mean_abs_error")),
        "latency_ratio": _optional_float(summary.get("average_carved_vs_dense_latency_ratio")),
        "average_dense_latency_s": _optional_float(summary.get("average_dense_latency_s")),
        "average_carved_latency_s": _optional_float(summary.get("average_carved_latency_s")),
        "worst_layer": summary.get("worst_layer"),
        "worst_layer_selected_vs_all_max_abs_error": _optional_float(
            summary.get("worst_layer_selected_vs_all_max_abs_error")
        ),
        "active_experts": active_experts,
        "warning_count": len(_list(payload.get("warnings"))),
    }


def _summarize_active_experts(report: dict[str, Any]) -> list[dict[str, Any]]:
    by_layer: dict[int, set[tuple[int, ...]]] = {}
    for item in _list(report.get("active_experts")):
        if not isinstance(item, dict):
            continue
        layer = _optional_int(item.get("layer"))
        if layer is None:
            continue
        experts = tuple(
            expert
            for expert in (_optional_int(expert) for expert in _list(item.get("experts")))
            if expert is not None
        )
        by_layer.setdefault(layer, set()).add(experts)
    return [
        {
            "layer": layer,
            "expert_sets": [list(experts) for experts in sorted(sets)],
        }
        for layer, sets in sorted(by_layer.items())
    ]


def _comparison_warnings(reports: list[dict[str, Any]]) -> list[str]:
    warnings = []
    modes = {mode for report in reports for mode in report.get("expert_modes", [])}
    if modes - {"all"}:
        warnings.append("Subset modes compare quality/speed tradeoffs against dense outputs and all-expert attribution.")
    if any(not report.get("passed") for report in reports):
        warnings.append("At least one report failed its eval threshold; inspect max/mean error before ranking by speed.")
    models = {str(report.get("model")) for report in reports}
    if len(models) > 1:
        warnings.append("Reports reference more than one model; rank comparisons may mix different baselines.")
    return warnings


def _comparison_summary(comparison: dict[str, Any]) -> str:
    best = _dict(comparison.get("best"))
    fastest = _dict(comparison.get("fastest"))
    lowest_error = _dict(comparison.get("lowest_error"))
    cards = [
        ("Reports", comparison.get("report_count")),
        ("Quality-First Best", best.get("label")),
        ("Lowest Error", lowest_error.get("label")),
        ("Lowest Max Abs", lowest_error.get("max_abs_error")),
        ("Fastest Label", fastest.get("label")),
        ("Fastest Ratio", fastest.get("latency_ratio")),
    ]
    return f"""
<section>
  <h2>Summary</h2>
  <p class="subtle">{escape(_text(comparison.get("ranking_policy")))}</p>
  <div class="cards">
    {"".join(_card(label, value) for label, value in cards)}
  </div>
</section>
"""


def _comparison_warnings_section(comparison: dict[str, Any]) -> str:
    warnings = [str(item) for item in _list(comparison.get("warnings"))]
    if not warnings:
        return ""
    return f"""
<section>
  <h2>Warnings And Assumptions</h2>
  <ul class="warnings">
    {"".join(f"<li>{escape(item)}</li>" for item in warnings)}
  </ul>
</section>
"""


def _comparison_table(comparison: dict[str, Any]) -> str:
    rows = []
    for item in _list(comparison.get("ranked")):
        if not isinstance(item, dict):
            continue
        rows.append(
            "<tr>"
            f"<td>{escape(_text(item.get('rank')))}</td>"
            f"<td>{escape(_text(item.get('label')))}</td>"
            f"<td>{escape(', '.join(str(mode) for mode in _list(item.get('expert_modes'))))}</td>"
            f"<td>{escape(_text(item.get('passed')))}</td>"
            f"<td>{escape(_number(item.get('max_abs_error')))}</td>"
            f"<td>{escape(_number(item.get('mean_abs_error')))}</td>"
            f"<td>{escape(_number(item.get('latency_ratio')))}</td>"
            f"<td>{escape(_text(item.get('worst_layer')))}</td>"
            f"<td>{escape(_number(item.get('worst_layer_selected_vs_all_max_abs_error')))}</td>"
            "</tr>"
        )
    return _section_table(
        title="Ranked Reports",
        headers=[
            "Rank",
            "Label",
            "Modes",
            "Passed",
            "Max Abs",
            "Mean Abs",
            "Latency Ratio",
            "Worst Layer",
            "Worst Layer Delta",
        ],
        rows=rows,
    )


def _comparison_experts(comparison: dict[str, Any]) -> str:
    rows = []
    for report in _list(comparison.get("ranked")):
        if not isinstance(report, dict):
            continue
        for item in _list(report.get("active_experts")):
            if not isinstance(item, dict):
                continue
            expert_sets = " ".join(_expert_chips(experts) for experts in _list(item.get("expert_sets")))
            rows.append(
                "<tr>"
                f"<td>{escape(_text(report.get('label')))}</td>"
                f"<td>{escape(_text(item.get('layer')))}</td>"
                f"<td>{expert_sets}</td>"
                "</tr>"
            )
    return _section_table(
        title="Active Expert Sets",
        headers=["Report", "Layer", "Expert Sets"],
        rows=rows,
    )


def _summary_cards(report: dict[str, Any]) -> str:
    summary = _dict(report.get("summary"))
    cards = [
        ("Samples", report.get("sample_count")),
        ("Max Abs Error", report.get("max_abs_error")),
        ("Mean Abs Error", report.get("mean_abs_error")),
        ("Avg Dense Latency", summary.get("average_dense_latency_s")),
        ("Avg Carved Latency", summary.get("average_carved_latency_s")),
        ("Latency Ratio", summary.get("average_carved_vs_dense_latency_ratio")),
        ("Worst Sample", summary.get("worst_sample_index")),
        ("Worst Layer", summary.get("worst_layer")),
    ]
    return f"""
<section>
  <h2>Summary</h2>
  <div class="cards">
    {"".join(_card(label, value) for label, value in cards)}
  </div>
</section>
"""


def _warnings(report: dict[str, Any]) -> str:
    warnings = [str(item) for item in _list(report.get("warnings"))]
    modes = {str(sample.get("expert_mode")) for sample in _list(report.get("samples")) if isinstance(sample, dict)}
    if modes - {"all"}:
        warnings.append("Selected-expert modes are quality/speed tradeoff probes against dense and all-expert carved outputs.")
    if not warnings:
        return ""
    return f"""
<section>
  <h2>Warnings And Assumptions</h2>
  <ul class="warnings">
    {"".join(f"<li>{escape(item)}</li>" for item in warnings)}
  </ul>
</section>
"""


def _samples_table(report: dict[str, Any]) -> str:
    rows = []
    for sample in _list(report.get("samples")):
        if not isinstance(sample, dict):
            continue
        rows.append(
            "<tr>"
            f"<td>{escape(_text(sample.get('index')))}</td>"
            f"<td>{escape(_text(sample.get('source')))}</td>"
            f"<td>{escape(_text(sample.get('expert_mode')))}</td>"
            f"<td>{escape(_number(sample.get('max_abs_error')))}</td>"
            f"<td>{escape(_number(sample.get('mean_abs_error')))}</td>"
            f"<td>{escape(_number(sample.get('carved_vs_dense_latency_ratio')))}</td>"
            f"<td>{escape(_text(sample.get('allclose')))}</td>"
            "</tr>"
        )
    return _section_table(
        title="Samples",
        headers=["Index", "Source", "Mode", "Max Abs", "Mean Abs", "Latency Ratio", "Allclose"],
        rows=rows,
    )


def _active_experts(report: dict[str, Any]) -> str:
    rows = []
    for item in _list(report.get("active_experts")):
        if not isinstance(item, dict):
            continue
        rows.append(
            "<tr>"
            f"<td>{escape(_text(item.get('sample_index')))}</td>"
            f"<td>{escape(_text(item.get('layer')))}</td>"
            f"<td>{escape(_text(item.get('mode')))}</td>"
            f"<td>{_expert_chips(item.get('experts'))}</td>"
            "</tr>"
        )
    return _section_table(
        title="Active Experts",
        headers=["Sample", "Layer", "Mode", "Experts"],
        rows=rows,
    )


def _layer_attribution(report: dict[str, Any]) -> str:
    rows = []
    for item in _list(report.get("layer_attribution")):
        if not isinstance(item, dict):
            continue
        rows.append(
            "<tr>"
            f"<td>{escape(_text(item.get('sample_index')))}</td>"
            f"<td>{escape(_text(item.get('layer')))}</td>"
            f"<td>{_expert_chips(item.get('experts'))}</td>"
            f"<td>{escape(_number(item.get('dense_vs_all_max_abs_error')))}</td>"
            f"<td>{escape(_number(item.get('dense_vs_selected_max_abs_error')))}</td>"
            f"<td>{escape(_number(item.get('selected_vs_all_max_abs_error')))}</td>"
            f"<td>{escape(_number(item.get('selected_vs_all_mean_abs_error')))}</td>"
            "</tr>"
        )
    return _section_table(
        title="Layer Attribution",
        headers=[
            "Sample",
            "Layer",
            "Experts",
            "Dense-All Max",
            "Dense-Selected Max",
            "Selected-All Max",
            "Selected-All Mean",
        ],
        rows=rows,
    )


def _metadata(report: dict[str, Any]) -> str:
    package = _dict(report.get("package"))
    memory = _dict(report.get("memory"))
    replacements = _dict(report.get("replacements"))
    rows = [
        ("Source Model", report.get("source_model")),
        ("Adapter Family", report.get("adapter_family")),
        ("Package Expert Count", package.get("expert_count")),
        ("Dense Parameters", memory.get("dense_parameter_count")),
        ("Carved Parameters", memory.get("carved_parameter_count")),
        ("Carved Buffers", memory.get("carved_buffer_count")),
        ("Replaced Modules", len(_list(replacements.get("replaced")))),
    ]
    body = "".join(f"<tr><th>{escape(label)}</th><td>{escape(_text(value))}</td></tr>" for label, value in rows)
    return f"""
<section>
  <h2>Metadata</h2>
  <table><tbody>{body}</tbody></table>
</section>
"""


def _section_table(*, title: str, headers: list[str], rows: list[str]) -> str:
    if not rows:
        return ""
    header_html = "".join(f"<th>{escape(header)}</th>" for header in headers)
    return f"""
<section>
  <h2>{escape(title)}</h2>
  <div class="table-wrap">
    <table>
      <thead><tr>{header_html}</tr></thead>
      <tbody>{"".join(rows)}</tbody>
    </table>
  </div>
</section>
"""


def _card(label: str, value: Any) -> str:
    return f"""
<article class="card">
  <div class="label">{escape(label)}</div>
  <div class="value">{escape(_number(value))}</div>
</article>
"""


def _expert_chips(value: Any) -> str:
    return "".join(f'<span class="chip">E{escape(_text(expert))}</span>' for expert in _list(value))


def _number(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (float, int)):
        if isinstance(value, float):
            return f"{value:.6g}"
        return str(value)
    return str(value)


def _text(value: Any) -> str:
    if value is None:
        return "n/a"
    return str(value)


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _rank_number(value: Any) -> float:
    number = _optional_float(value)
    return number if number is not None else float("inf")


def _css() -> str:
    return """
:root { color-scheme: light; font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
body { margin: 0; background: #f6f7f9; color: #1d2430; }
.page { max-width: 1180px; margin: 0 auto; padding: 32px 20px 56px; }
.header { display: flex; justify-content: space-between; gap: 24px; align-items: flex-start; margin-bottom: 28px; }
.eyebrow { margin: 0 0 8px; color: #516175; font-size: 13px; font-weight: 700; text-transform: uppercase; letter-spacing: 0; }
h1 { margin: 0; font-size: 28px; line-height: 1.2; }
h2 { margin: 28px 0 12px; font-size: 18px; }
.subtle { color: #687589; margin: 8px 0 0; font-size: 14px; overflow-wrap: anywhere; }
.status { padding: 8px 12px; border-radius: 6px; font-weight: 700; text-transform: uppercase; font-size: 13px; }
.status.passed { background: #dff6e7; color: #12652f; }
.status.review { background: #fff1cf; color: #805600; }
.cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 12px; }
.card { background: #fff; border: 1px solid #d8dee8; border-radius: 8px; padding: 14px; }
.label { color: #64748b; font-size: 12px; font-weight: 700; text-transform: uppercase; letter-spacing: 0; }
.value { margin-top: 6px; font-size: 20px; font-weight: 750; overflow-wrap: anywhere; }
.warnings { background: #fff; border: 1px solid #ecd898; border-radius: 8px; padding: 14px 18px 14px 34px; }
.table-wrap { overflow-x: auto; background: #fff; border: 1px solid #d8dee8; border-radius: 8px; }
table { width: 100%; border-collapse: collapse; font-size: 14px; }
th, td { padding: 10px 12px; border-bottom: 1px solid #e7ebf0; text-align: left; vertical-align: top; }
thead th { background: #edf1f5; color: #334155; font-size: 12px; text-transform: uppercase; letter-spacing: 0; }
tbody tr:last-child td, tbody tr:last-child th { border-bottom: 0; }
.chip { display: inline-block; margin: 0 4px 4px 0; padding: 3px 7px; border-radius: 999px; background: #e7eefc; color: #1d4f91; font-size: 12px; font-weight: 700; }
@media (max-width: 700px) { .header { display: block; } .status { display: inline-block; margin-top: 16px; } }
"""
