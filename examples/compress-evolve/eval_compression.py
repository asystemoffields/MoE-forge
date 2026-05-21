"""L0 verifier for the compress-evolve loop -- MULTI-AXIS "good".

A method is good in more than one way, so the verifier scores several axes and the QD archive
keeps the multi-objective Pareto frontier over them (an idea good on ANY axis is caught, without a
human weighting them):
  - shipped bytes (len(artifact))            -- compression
  - held-out NLL, mean over domains          -- quality
  - worst-DOMAIN NLL delta (prose/code/knowledge) -- robustness (catches calibration-overfit)
  - worst-case TAIL delta (max per-sequence) -- no catastrophic inputs
  - decode seconds (time of decompress)      -- load-time cost

decompress sees ONLY the artifact (Hutter rule); the held-out domains are disjoint from the
calib set compress may study; quality is measured through the real decode + forward path.

Contract (unchanged):
    def compress(model, calib_tokens, budget_bytes) -> bytes
    def decompress(artifact: bytes) -> nn.Module
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path

# Calibration text the candidate may study.
CALIB = [
    "The mitochondria is the powerhouse of the cell, converting nutrients into usable energy.",
    "In 1969, Apollo 11 landed the first humans on the Moon after a four-day journey.",
    "def quicksort(xs):\n    if len(xs) <= 1:\n        return xs\n    pivot = xs[0]\n    rest = xs[1:]",
    "The capital of France is Paris, a city long known for its art, food, and architecture.",
    "Photosynthesis converts carbon dioxide and water into glucose using energy from sunlight.",
    "She argued that the proposed economic policy would raise inflation in the short term.",
    "Water boils at one hundred degrees Celsius at standard atmospheric pressure at sea level.",
    "For a right triangle, the square of the hypotenuse equals the sum of the squares of the legs.",
    "Once upon a time, in a village nestled between two tall mountains, lived a curious child.",
    "Quantum entanglement links the measured states of two particles regardless of the distance.",
    "The recipe calls for two cups of flour, a teaspoon of salt, three eggs, and a cup of milk.",
    "Markets fell sharply on news of rising interest rates and weaker corporate earnings reports.",
    "A compiler translates source code written by a programmer into machine instructions.",
    "The treaty was signed after months of negotiation between the two neighboring countries.",
    "Photovoltaic panels convert sunlight directly into electricity through the photoelectric effect.",
]

# Held-out, split by DOMAIN (disjoint from CALIB). Robustness = the worst domain, not the average.
HELDOUT_DOMAINS = {
    "prose": [
        "The river wound slowly through the valley, reflecting the orange light of the setting sun.",
        "The library closed early on Sunday, so the students moved to a nearby cafe to keep studying.",
        "The merchant counted his coins twice before locking the heavy chest for the night.",
        "The novel explores themes of memory, loss, and the slow passage of an ordinary life.",
        "Shakespeare wrote both comedies and tragedies during the late sixteenth century.",
        "She folded the letter carefully and placed it beneath a stack of yellowing photographs.",
    ],
    "code": [
        "import numpy as np\n\ndef mean(values):\n    return sum(values) / len(values)",
        "for i in range(len(items)):\n    if items[i] is None:\n        items[i] = default_value",
        "class Stack:\n    def __init__(self):\n        self._data = []\n    def push(self, x):\n        self._data.append(x)",
        "def fib(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a",
        "with open(path, 'r') as f:\n    lines = [line.strip() for line in f if line.strip()]",
        "result = sorted(records, key=lambda r: (r['score'], -r['age']), reverse=True)",
    ],
    "knowledge": [
        "The human heart pumps roughly five liters of blood through the body every single minute.",
        "Gravity causes objects with mass to attract one another in proportion to their masses.",
        "A neural network adjusts its weights by propagating errors backward through its layers.",
        "Vaccines train the immune system to recognize and fight specific pathogens more quickly.",
        "Electrons occupy discrete energy levels around the nucleus of an atom.",
        "Distributed systems must tolerate the failure of individual machines without losing data.",
    ],
}


def load_candidate(path: Path):
    spec = importlib.util.spec_from_file_location("candidate", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for name in ("compress", "decompress"):
        if not hasattr(module, name):
            raise ValueError(f"candidate must define {name}(...)")
    return module.compress, module.decompress


def make_batches(tokenizer, texts, seq_len, torch):
    ids = tokenizer("\n\n".join(texts), return_tensors="pt").input_ids[0]
    usable = (ids.shape[0] // seq_len) * seq_len
    if usable == 0:
        return ids.unsqueeze(0)
    return ids[:usable].reshape(-1, seq_len)


def seq_nlls(model, batches, torch):
    """Per-sequence NLL (one value per chunk) so we can take both mean and worst-case tail."""
    out = []
    with torch.no_grad():
        for i in range(batches.shape[0]):
            chunk = batches[i : i + 1]
            out.append(float(model(input_ids=chunk, labels=chunk).loss))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--model", default="HuggingFaceTB/SmolLM-135M")
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--budget-bytes", type=int, default=0)
    args = parser.parse_args()

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        compress, decompress = load_candidate(args.candidate)
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForCausalLM.from_pretrained(args.model).eval()

        calib = make_batches(tokenizer, CALIB, args.seq_len, torch)
        domains = {d: make_batches(tokenizer, texts, args.seq_len, torch) for d, texts in HELDOUT_DOMAINS.items()}
        base = {d: seq_nlls(model, b, torch) for d, b in domains.items()}
        full_bytes = sum(p.numel() * p.element_size() for p in model.parameters())

        artifact = compress(model, calib, args.budget_bytes)
        if not isinstance(artifact, (bytes, bytearray)):
            print(json.dumps({"error": f"compress must return bytes, got {type(artifact).__name__}"}))
            return
        shipped = len(artifact)

        t0 = time.perf_counter()
        rebuilt = decompress(bytes(artifact)).eval()
        decode_seconds = time.perf_counter() - t0
        # Resident bytes = what the runnable model HOLDS (the real local-deploy memory + bandwidth/
        # speed proxy). Compute-in-compressed-form methods hold packed weights -> small; methods that
        # dequantize to fp32 hold the full model -> large. This is the axis that matters for the mission.
        resident_bytes = sum(t.numel() * t.element_size() for t in rebuilt.state_dict().values())
        meth = {d: seq_nlls(rebuilt, b, torch) for d, b in domains.items()}

        base_mean = {d: sum(v) / len(v) for d, v in base.items()}
        meth_mean = {d: sum(v) / len(v) for d, v in meth.items()}
        delta_by_domain = {d: round(meth_mean[d] - base_mean[d], 4) for d in domains}
        all_base = [x for v in base.values() for x in v]
        all_meth = [x for v in meth.values() for x in v]
        nll = sum(all_meth) / len(all_meth)
        baseline_nll = sum(all_base) / len(all_base)
        worst_domain_delta = max(delta_by_domain.values())
        tail_delta = max(m - b for d in domains for m, b in zip(meth[d], base[d]))

        result = {
            "bytes": shipped,
            "full_bytes": int(full_bytes),
            "ratio": round(full_bytes / max(shipped, 1), 3),
            "resident_bytes": int(resident_bytes),
            "resident_ratio": round(full_bytes / max(resident_bytes, 1), 3),
            "nll": round(nll, 4),
            "baseline_nll": round(baseline_nll, 4),
            "nll_delta": round(nll - baseline_nll, 4),
            "nll_delta_by_domain": delta_by_domain,
            "worst_domain_delta": round(worst_domain_delta, 4),
            "tail_delta": round(tail_delta, 4),
            "decode_seconds": round(decode_seconds, 3),
        }
        if args.budget_bytes and shipped > args.budget_bytes:
            result["over_budget"] = True
        print(json.dumps(result, indent=2))
    except Exception as exc:
        import traceback
        print(json.dumps({"error": str(exc), "trace": traceback.format_exc()[-700:]}))


if __name__ == "__main__":
    main()
