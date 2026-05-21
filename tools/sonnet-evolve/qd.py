"""qd.py - the algorithmic search core for the compress-evolve loop (MAP-Elites + Pareto frontier).

This is what removes the human from the direction-choosing. Instead of a person prescribing
"try QuaRot / try SpQR", a persistent archive bins every evaluated method by behavior
(compression-ratio band x self-reported method family), keeps the best per cell, and emits an
"exploration brief": the current rate-distortion frontier, the families already tried, and the
EMPTY/under-explored cells. A generator (LLM, via API or subagent) is handed that brief and asked
to fill an empty cell or beat a frontier point with a method of ITS OWN choosing. The archive +
the empty-cell pressure are the search operator; the human is no longer the idea source.

CLI:
  python tools/sonnet-evolve/qd.py add   --archive A.json --name N --family F --gen G \
         --result <eval_compression.json> --candidate <path>
  python tools/sonnet-evolve/qd.py brief --archive A.json     # the prompt context for the next generation
  python tools/sonnet-evolve/qd.py show  --archive A.json     # frontier + all cells
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

# Compression-ratio bands (vs the model's native bytes). Edges in x.
RATIO_EDGES = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 8.0, 1e9]


def ratio_band(ratio: float) -> str:
    for lo, hi in zip(RATIO_EDGES, RATIO_EDGES[1:]):
        if lo <= ratio < hi:
            return f"{lo:g}-{hi:g}x" if hi < 1e9 else f">{lo:g}x"
    return ">8x"


def load(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"format": "compress_evolve_qd_archive", "cells": {}, "history": [], "rejected": []}


def save(path: Path, archive: dict) -> None:
    path.write_text(json.dumps(archive, indent=2) + "\n", encoding="utf-8")


def insert(archive: dict, record: dict) -> tuple[bool, str]:
    """Insert a record; keep the best (lowest nll) per (band x family) cell. Returns (is_new_best, cell)."""
    band = ratio_band(record["ratio"])
    cell = f"{band}|{record['family']}"
    archive.setdefault("history", []).append(record)
    cells = archive.setdefault("cells", {})
    cur = cells.get(cell)
    improved = cur is None or record["nll"] < cur["nll"]
    if improved:
        cells[cell] = record
    return improved, cell


FRONTIER_OBJECTIVES = ("resident_bytes", "nll_delta", "worst_domain_delta", "decode_seconds")


def _obj(rec: dict, key: str) -> float:
    v = rec.get(key)
    return float(v) if isinstance(v, (int, float)) else float("inf")


def frontier(archive: dict, objectives=FRONTIER_OBJECTIVES) -> list[dict]:
    """Multi-objective Pareto frontier, ALL objectives minimized: bytes, NLL-delta, worst-domain
    delta (robustness), decode seconds. A point is dominated iff another is <= on every objective
    and < on at least one. Missing axes count as worst (+inf), so old single-axis records must be
    re-evaluated to qualify. This is how 'good' widens without a human weighting the axes."""
    pts = list(archive.get("cells", {}).values())
    front = []
    for p in pts:
        dominated = False
        for q in pts:
            if q is p:
                continue
            if all(_obj(q, k) <= _obj(p, k) for k in objectives) and any(_obj(q, k) < _obj(p, k) for k in objectives):
                dominated = True
                break
        if not dominated:
            front.append(p)
    return sorted(front, key=lambda r: r.get("bytes", 0))


def occupied_bands(archive: dict) -> set[str]:
    return {c.split("|", 1)[0] for c in archive.get("cells", {})}


def families(archive: dict) -> set[str]:
    return {c.split("|", 1)[1] for c in archive.get("cells", {})}


def brief(archive: dict) -> str:
    cells = archive.get("cells", {})
    if not cells:
        return ("ARCHIVE EMPTY. Propose any complete compression method (you choose the technique) "
                "and tag its family.")
    fr = frontier(archive)
    occ = occupied_bands(archive)
    all_bands = [f"{lo:g}-{hi:g}x" if hi < 1e9 else f">{lo:g}x"
                 for lo, hi in zip(RATIO_EDGES, RATIO_EDGES[1:])]
    empty_bands = [b for b in all_bands if b not in occ]
    fams = sorted(families(archive))

    lines = ["FRONTIER (minimize ALL: RESIDENT bytes = RAM-to-run [the key local-deploy axis, also the "
             "bandwidth/speed proxy], NLL-delta, worst-domain delta=robustness, decode seconds; disk shown for ref):"]
    for p in fr:
        lines.append(f"  - resident {p.get('resident_ratio', '?')}x  NLL{p['nll_delta']:+.3f}  "
                     f"worst-domain{_obj(p, 'worst_domain_delta'):+.3f}  decode {_obj(p, 'decode_seconds'):.1f}s  "
                     f"(disk {p['ratio']:.2f}x)  [{p['family']}]  ({p['name']})")
    lines.append("")
    lines.append(f"FAMILIES ALREADY TRIED ({len(fams)}): {', '.join(fams)}")
    lines.append(f"EMPTY COMPRESSION-RATIO BANDS (no method here yet): {', '.join(empty_bands) or 'none'}")
    lines.append("")
    lines.append("YOUR JOB: propose a method that EITHER (a) lands in an empty band above, OR "
                 "(b) Pareto-beats a frontier point on ANY axis -- lower RESIDENT memory (the model must "
                 "RUN holding weights compressed, NOT dequantize to fp32), lower NLL, more robust across "
                 "prose/code/knowledge, or faster to decode -- using a technique from a family NOT in the "
                 "tried list, or a novel combination. The big prize: match the best NLL while running in "
                 "low resident memory. Choose the approach yourself; do not reuse an existing family "
                 "unless you change it fundamentally. Tag your method with a short family name.")
    return "\n".join(lines)


def preserve(archive: dict, repo_root: Path, dst_dir: Path) -> int:
    """Copy the SOURCE of every current frontier method into dst_dir (durable + committable) and
    write FRONTIER.json. Frontier methods often come from gitignored scratch files the loop churns,
    so this is how a good idea is actually CAUGHT and kept -- the code, not just its score."""
    fr = frontier(archive)
    dst_dir.mkdir(parents=True, exist_ok=True)
    kept = 0
    for p in fr:
        raw = p.get("candidate_path", "")
        if not raw:
            continue
        src = Path(raw)
        if not src.is_absolute():
            src = repo_root / raw
        if src.exists():
            try:
                shutil.copyfile(src, dst_dir / f"{p['name']}.py")
                kept += 1
            except OSError:
                pass
    (dst_dir / "FRONTIER.json").write_text(json.dumps(fr, indent=2) + "\n", encoding="utf-8")
    return kept


def audit(archive: dict, recent: int = 2) -> str:
    """Health report for a supervisor that stops the loop to see what's happening:
    frontier, family diversity, failure rate, and stall/narrowing detection."""
    cells = archive.get("cells", {})
    if not cells:
        return "AUDIT: archive empty (nothing evaluated yet)."
    hist = archive.get("history", [])
    rej = archive.get("rejected", [])
    fr = frontier(archive)
    fams = sorted(families(archive))
    occ = occupied_bands(archive)
    all_bands = [f"{lo:g}-{hi:g}x" if hi < 1e9 else f">{lo:g}x"
                 for lo, hi in zip(RATIO_EDGES, RATIO_EDGES[1:])]
    empty = [b for b in all_bands if b not in occ]

    def gnum(g):
        try:
            return int(g)
        except (TypeError, ValueError):
            return -1

    maxg = max((gnum(r.get("generation")) for r in hist), default=-1)
    recent_front = [p for p in fr if gnum(p.get("generation")) > maxg - recent] if maxg >= 0 else fr
    n_arch, n_rej = len(hist), len(rej)
    rej_rate = n_rej / max(n_arch + n_rej, 1)

    lines = ["=== COMPRESS-EVOLVE AUDIT ==="]
    lines.append(f"evaluated: {n_arch} archived, {n_rej} rejected ({rej_rate:.0%} reject rate)")
    lines.append(f"families ({len(fams)}): {', '.join(fams)}")
    lines.append(f"empty ratio-bands: {', '.join(empty) or 'none'}")
    lines.append("frontier (rate-distortion):")
    for p in fr:
        lines.append(f"  {p['ratio']:.2f}x  NLL{p['nll_delta']:+.3f}  worst-domain{_obj(p, 'worst_domain_delta'):+.3f}  "
                     f"decode {_obj(p, 'decode_seconds'):.1f}s  [{p['family']}]  gen={p.get('generation')}  ({p['name']})")

    flags = []
    if maxg >= 0 and not recent_front:
        flags.append(f"STALLED: no frontier point from the last {recent} generation(s)")
    if len(fams) <= 2 and n_arch >= 6:
        flags.append("NARROWING: <=2 families; inject new conceptions")
    if rej_rate > 0.6 and (n_arch + n_rej) >= 6:
        flags.append("HIGH-REJECT: >60% of candidates fail; clarify/loosen constraints")
    lines.append("VERDICT: " + (" | ".join(flags) if flags else "HEALTHY (frontier advancing, families diversifying)"))
    lines.append("SUPERVISOR TODO: re-verify the frontier leader on MORE held-out tokens (Goodhart/noise "
                 "check); read the top methods to confirm the win is mechanistic, not metric-gaming; then "
                 "decide continue / adjust the prompt / inject diversity / stop.")
    return "\n".join(lines)


def _record_from_result(args) -> dict:
    res = json.loads(Path(args.result).read_text(encoding="utf-8"))
    return {
        "name": args.name,
        "family": args.family,
        "generation": int(args.gen),
        "candidate_path": args.candidate,
        "bytes": int(res.get("bytes", 0)),
        "full_bytes": int(res.get("full_bytes", 0)),
        "ratio": float(res.get("ratio", 0.0)),
        "nll": float(res.get("nll", float("inf"))) if res.get("nll") is not None else float("inf"),
        "baseline_nll": float(res.get("baseline_nll", 0.0)),
        "nll_delta": float(res.get("nll_delta", float("inf"))) if res.get("nll_delta") is not None else float("inf"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("add")
    a.add_argument("--archive", type=Path, required=True)
    a.add_argument("--name", required=True)
    a.add_argument("--family", required=True)
    a.add_argument("--gen", default="0")
    a.add_argument("--result", type=Path, required=True)
    a.add_argument("--candidate", default="")

    b = sub.add_parser("brief")
    b.add_argument("--archive", type=Path, required=True)

    s = sub.add_parser("show")
    s.add_argument("--archive", type=Path, required=True)

    au = sub.add_parser("audit")
    au.add_argument("--archive", type=Path, required=True)

    pr = sub.add_parser("preserve")
    pr.add_argument("--archive", type=Path, required=True)
    pr.add_argument("--dir", type=Path, default=REPO / "examples" / "compress-evolve" / "frontier")

    args = parser.parse_args()
    archive = load(args.archive)

    if args.cmd == "add":
        rec = _record_from_result(args)
        if rec["nll"] != rec["nll"] or rec["nll"] == float("inf"):  # NaN/inf -> failed, do not enshrine
            print(json.dumps({"inserted": False, "reason": "non-finite nll (failed candidate)", "name": rec["name"]}))
            return
        improved, cell = insert(archive, rec)
        save(args.archive, archive)
        print(json.dumps({"inserted": True, "new_best_in_cell": improved, "cell": cell,
                          "ratio": rec["ratio"], "nll_delta": rec["nll_delta"]}))
    elif args.cmd == "brief":
        print(brief(archive))
    elif args.cmd == "show":
        fr = frontier(archive)
        print(f"cells: {len(archive.get('cells', {}))}  history: {len(archive.get('history', []))}")
        print("frontier:")
        for p in fr:
            print(f"  {p['ratio']:.2f}x  NLL+{p['nll_delta']:+.3f}  [{p['family']}]  {p['name']}")
    elif args.cmd == "audit":
        print(audit(archive))
    elif args.cmd == "preserve":
        n = preserve(archive, REPO, args.dir)
        print(f"preserved {n} frontier methods -> {args.dir}")


if __name__ == "__main__":
    main()
