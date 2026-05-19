from __future__ import annotations

from html import escape
import json
from pathlib import Path
from typing import Any


class ReportError(RuntimeError):
    """Raised when a report artifact cannot be rendered."""


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
