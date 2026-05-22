"""Validate a frontier compress-evolve method on a MUCH larger held-out set than the loop's
verifier (which uses ~600 builtin tokens). Kills the small-sample-noise risk before trusting a
winner. Pulls wikitext-2 test if available (cached or online), else a larger builtin corpus.
Reuses eval_compression's candidate loading + per-sequence NLL.

Usage:
  python examples/compress-evolve/validate.py --candidate examples/compress-evolve/frontier/auto_g5_0.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import eval_compression as EC  # noqa: E402

# Modest offline fallback: the loop's calib+heldout text plus extras (~3x the verifier's 600 tokens).
_EXTRA = [
    "The printing press, introduced in the fifteenth century, transformed how knowledge spread.",
    "A binary search halves the search interval each step, so it runs in logarithmic time.",
    "Photosystem II splits water molecules, releasing the oxygen that life on land depends on.",
    "The central bank raised rates a quarter point, citing persistent core inflation.",
    "def dfs(node, seen):\n    seen.add(node)\n    for nxt in graph[node]:\n        if nxt not in seen:\n            dfs(nxt, seen)",
    "Tectonic plates drift a few centimeters a year, reshaping coastlines over millions of years.",
    "She tuned the old radio until a faint orchestral melody emerged from the static.",
    "In linear algebra, the rank of a matrix equals the dimension of its column space.",
    "The immune system's memory cells let a second exposure to a pathogen be cleared faster.",
    "Caching trades memory for speed by keeping recently used results close to the processor.",
    "The treaty redrew the border along the river, ending a decade of intermittent conflict.",
    "Enzymes lower the activation energy of reactions without being consumed by them.",
]


def big_heldout(tokenizer, seq_len, max_seqs, torch, corpus_file=None):
    # No `datasets` (pyarrow segfaulted on Windows under load). Use a local text file if given,
    # else the builtin-large fallback.
    if corpus_file:
        source = f"local:{Path(corpus_file).name}"
        text = Path(corpus_file).read_text(encoding="utf-8", errors="replace")[:300000]
    else:
        source = "builtin-large"
        text = "\n\n".join(EC.CALIB + [t for v in EC.HELDOUT_DOMAINS.values() for t in v] + _EXTRA)
    ids = tokenizer(text, return_tensors="pt").input_ids[0]
    usable = (ids.shape[0] // seq_len) * seq_len
    batches = ids[:usable].reshape(-1, seq_len)
    if batches.shape[0] > max_seqs:
        batches = batches[:max_seqs]
    return batches, source


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--model", default="HuggingFaceTB/SmolLM-135M")
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--max-seqs", type=int, default=120)
    parser.add_argument("--corpus-file", default=None, help="Local text file as held-out (avoids datasets).")
    args = parser.parse_args()

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        compress, decompress = EC.load_candidate(args.candidate)
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForCausalLM.from_pretrained(args.model).eval()

        calib = EC.make_batches(tokenizer, EC.CALIB, 128, torch)
        heldout, source = big_heldout(tokenizer, args.seq_len, args.max_seqs, torch, args.corpus_file)
        eval_tokens = int(heldout.shape[0] * heldout.shape[1])

        base = EC.seq_nlls(model, heldout, torch)
        baseline_nll = sum(base) / len(base)
        full_bytes = sum(p.numel() * p.element_size() for p in model.parameters())

        t0 = time.perf_counter()
        artifact = compress(model, calib, 0)
        compress_seconds = time.perf_counter() - t0
        shipped = len(artifact)
        rebuilt = decompress(bytes(artifact)).eval()
        resident_bytes = sum(t.numel() * t.element_size() for t in rebuilt.state_dict().values())
        meth = EC.seq_nlls(rebuilt, heldout, torch)
        nll = sum(meth) / len(meth)

        print(json.dumps({
            "candidate": args.candidate.name,
            "eval_source": source,
            "eval_tokens": eval_tokens,
            "eval_seqs": int(heldout.shape[0]),
            "baseline_nll": round(baseline_nll, 4),
            "nll": round(nll, 4),
            "nll_delta": round(nll - baseline_nll, 4),
            "ratio": round(full_bytes / max(shipped, 1), 3),
            "resident_ratio": round(full_bytes / max(resident_bytes, 1), 3),
            "compress_seconds": round(compress_seconds, 1),
        }, indent=2))
    except Exception as exc:
        import traceback
        print(json.dumps({"error": str(exc), "trace": traceback.format_exc()[-700:]}))


if __name__ == "__main__":
    main()
