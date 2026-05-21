"""autoloop.py - one reusable step of the autonomous compress-evolve loop.

Given candidate method files, run each through the hermetic verifier (eval_compression.py) under
a wall-clock compute cap, then insert the finite results into the QD archive (qd.py). This is the
"evaluator disposes + archive records" half of the loop; the generation half is an LLM reading
`qd.py brief` (so the archive, not a human, chooses directions).

Each candidate's family tag is read from a module-level `FAMILY = "..."` constant if present,
else inferred from the filename stem (so seed/baseline files self-describe).

Usage:
  python examples/compress-evolve/autoloop.py --archive examples/compress-evolve/archive.json \
      --gen 5 --timeout 150 <candidate1.py> <candidate2.py> ...
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools" / "sonnet-evolve"))
import qd  # noqa: E402

EVAL = "examples/compress-evolve/eval_compression.py"
_FAMILY_RE = re.compile(r"""^FAMILY\s*=\s*['"]([^'"]+)['"]""", re.MULTILINE)


def family_of(path: Path) -> str:
    text = path.read_text(encoding="utf-8", errors="replace")
    m = _FAMILY_RE.search(text)
    return m.group(1) if m else path.stem


def last_json(text: str):
    """Return the last top-level JSON object printed by the evaluator."""
    decoder = json.JSONDecoder()
    found = None
    i = 0
    while i < len(text):
        if text[i] != "{":
            i += 1
            continue
        try:
            val, end = decoder.raw_decode(text[i:])
            if isinstance(val, dict):
                found = val
            i += end
        except json.JSONDecodeError:
            i += 1
    return found


def evaluate(candidate: Path, repo: Path, timeout: int) -> dict:
    cmd = [sys.executable, EVAL, "--candidate", str(candidate)]
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "HF_HUB_DISABLE_PROGRESS_BARS": "1"}
    try:
        p = subprocess.run(cmd, cwd=str(repo), env=env, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"error": f"timeout >{timeout}s (over compute budget)"}
    payload = last_json(p.stdout)
    if payload is None:
        return {"error": "no JSON", "stderr": p.stderr[-300:]}
    return payload


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--archive", type=Path, required=True)
    ap.add_argument("--gen", default="auto")
    ap.add_argument("--timeout", type=int, default=150)
    ap.add_argument("candidates", nargs="+", type=Path)
    args = ap.parse_args()

    archive = qd.load(args.archive)
    repo = REPO

    for cand in args.candidates:
        fam = family_of(cand)
        res = evaluate(cand, repo, args.timeout)
        nll = res.get("nll")
        finite = isinstance(nll, (int, float)) and nll == nll and abs(nll) != float("inf")
        if "error" in res or not finite:
            reason = res.get("error", "non-finite nll")
            archive.setdefault("rejected", []).append(
                {"name": cand.stem, "family": fam, "generation": str(args.gen), "reason": reason})
            print(f"  REJECT  {cand.name:32} [{fam}]  {reason}")
            continue
        rec = {
            "name": cand.stem, "family": fam, "generation": str(args.gen),
            "candidate_path": str(cand),
            "bytes": int(res["bytes"]), "full_bytes": int(res.get("full_bytes", 0)),
            "ratio": float(res["ratio"]), "nll": float(res["nll"]),
            "resident_bytes": int(res.get("resident_bytes", res.get("full_bytes", 0))),
            "resident_ratio": float(res.get("resident_ratio", 1.0)),
            "baseline_nll": float(res.get("baseline_nll", 0.0)),
            "nll_delta": float(res.get("nll_delta", 0.0)),
            "worst_domain_delta": float(res.get("worst_domain_delta", res.get("nll_delta", 0.0))),
            "tail_delta": float(res.get("tail_delta", 0.0)),
            "decode_seconds": float(res.get("decode_seconds", 0.0)),
            "nll_delta_by_domain": res.get("nll_delta_by_domain", {}),
        }
        improved, cell = qd.insert(archive, rec)
        flag = "NEW-BEST" if improved else "kept-prev"
        print(f"  {flag:9} {cand.name:32} [{fam}]  {rec['ratio']:.2f}x  NLL{rec['nll_delta']:+.3f}  -> {cell}")

    qd.save(args.archive, archive)
    kept = qd.preserve(archive, REPO, REPO / "examples" / "compress-evolve" / "frontier")
    print(f"\n--- frontier (code of {kept} methods preserved to examples/compress-evolve/frontier/) ---")
    for p in qd.frontier(archive):
        print(f"  {p['ratio']:.2f}x  NLL{p['nll_delta']:+.3f}  [{p['family']}]  {p['name']}")


if __name__ == "__main__":
    main()
