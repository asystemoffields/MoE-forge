"""L0 verifier for the compress-evolve reconception loop.

A candidate is a COMPLETE compression method, free to reconceive what "compression" preserves:

    def compress(model, calib_tokens, budget_bytes) -> bytes      # the only per-model payload
    def decompress(artifact: bytes) -> nn.Module                  # gets ONLY the artifact

The candidate may run the model on calib tokens, study activations/gradients/whatever, and
decide its own internal objective. The verifier judges ONLY two frame-invariant quantities:
shipped bytes (len(artifact)) and held-out NLL of the reconstructed model. Because decompress
receives nothing but the artifact, any data it wants to keep costs bytes (the Hutter-prize
rule), and the held-out set is disjoint from calib, so a method cannot game the score by
memorizing or by leaking the eval set. That invariance is what lets the loop reconceive the
problem freely while staying honest.

Usage:
  python examples/compress-evolve/eval_compression.py --candidate seeds/int8_rtn.py
Prints JSON: {bytes, full_bytes, ratio, nll, baseline_nll, nll_delta}.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

# Calibration text the candidate may study (compress sees only these).
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

# Held-out text the score is measured on (disjoint from CALIB; decompress never sees it).
HELDOUT = [
    "The human heart pumps roughly five liters of blood through the body every single minute.",
    "Shakespeare wrote both comedies and tragedies during the late sixteenth and early seventeenth centuries.",
    "import numpy as np\n\ndef mean(values):\n    return sum(values) / len(values)",
    "The Great Barrier Reef is the largest living structure visible from space off the Australian coast.",
    "Inflation erodes the purchasing power of money, so a dollar buys less than it did before.",
    "Gravity causes objects with mass to attract one another in proportion to their masses.",
    "The library closed early on Sunday, so the students moved to a nearby cafe to keep studying.",
    "A neural network adjusts its weights by propagating errors backward through its layers.",
    "The river wound slowly through the valley, reflecting the orange light of the setting sun.",
    "Vaccines train the immune system to recognize and fight specific pathogens more quickly.",
    "The merchant counted his coins twice before locking the heavy chest for the night.",
    "Distributed systems must tolerate the failure of individual machines without losing data.",
    "Bees communicate the direction of food sources to the hive through a waggle dance.",
    "The novel explores themes of memory, loss, and the slow passage of an ordinary life.",
    "Electrons occupy discrete energy levels around the nucleus of an atom.",
]


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


def mean_nll(model, batches, torch):
    total, count = 0.0, 0
    with torch.no_grad():
        for i in range(batches.shape[0]):
            chunk = batches[i : i + 1]
            total += float(model(input_ids=chunk, labels=chunk).loss)
            count += 1
    return total / max(count, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--model", default="HuggingFaceTB/SmolLM-135M")
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--budget-bytes", type=int, default=0, help="0 = unconstrained (report frontier).")
    args = parser.parse_args()

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        compress, decompress = load_candidate(args.candidate)
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForCausalLM.from_pretrained(args.model).eval()

        calib = make_batches(tokenizer, CALIB, args.seq_len, torch)
        heldout = make_batches(tokenizer, HELDOUT, args.seq_len, torch)
        baseline = mean_nll(model, heldout, torch)
        full_bytes = sum(p.numel() * p.element_size() for p in model.parameters())

        artifact = compress(model, calib, args.budget_bytes)
        if not isinstance(artifact, (bytes, bytearray)):
            print(json.dumps({"error": f"compress must return bytes, got {type(artifact).__name__}"}))
            return
        shipped = len(artifact)

        rebuilt = decompress(bytes(artifact)).eval()
        quality = mean_nll(rebuilt, heldout, torch)

        result = {
            "bytes": shipped,
            "full_bytes": int(full_bytes),
            "ratio": round(full_bytes / max(shipped, 1), 3),
            "nll": round(quality, 4),
            "baseline_nll": round(baseline, 4),
            "nll_delta": round(quality - baseline, 4),
        }
        if args.budget_bytes and shipped > args.budget_bytes:
            result["over_budget"] = True
        print(json.dumps(result, indent=2))
    except Exception as exc:  # candidate crash / bad output -> worst score for the loop
        import traceback
        print(json.dumps({"error": str(exc), "trace": traceback.format_exc()[-700:]}))


if __name__ == "__main__":
    main()
