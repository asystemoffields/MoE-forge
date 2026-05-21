"""Capture one dense FFN layer's gated activations + down projection for grouping search.

Runs a few calibration prompts through the local dense model, hooks an FFN layer to grab
its input hidden states, and saves the per-channel gated activations `a = silu(h@gate^T)*(h@up^T)`
together with the down projection. That `.npz` is all `moeforge.grouping.oracle_topk_error`
needs, so the grouping search loop can then run with no model loads.

Usage:
  python examples/grouping-search/capture_layer.py \
      --source-model outputs/smollm-moe-release-v5/wrapper/source-model \
      --tokenizer outputs/smollm-moe-release-v5/wrapper \
      --layer 6 --max-tokens 600 --output examples/grouping-search/layer6.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

CALIBRATION = [
    "The mitochondria is the powerhouse of the cell, converting nutrients into usable energy.",
    "In 1969, Apollo 11 landed the first humans on the Moon after a four-day journey.",
    "def quicksort(xs):\n    if len(xs) <= 1:\n        return xs\n    pivot = xs[0]",
    "The capital of France is Paris, a city known for its art, food, and architecture.",
    "Photosynthesis converts carbon dioxide and water into glucose using sunlight.",
    "She argued that the economic policy would raise inflation in the short term.",
    "Water boils at 100 degrees Celsius at standard atmospheric pressure.",
    "The theorem states that for a right triangle, a squared plus b squared equals c squared.",
    "Once upon a time, in a village nestled between two mountains, lived a curious child.",
    "Quantum entanglement links the states of two particles regardless of distance.",
    "The recipe calls for two cups of flour, a teaspoon of salt, and three eggs.",
    "Markets fell sharply on news of rising interest rates and weaker earnings.",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-model", type=str, required=True,
                        help="HF hub id (e.g. HuggingFaceTB/SmolLM-135M) or local path.")
    parser.add_argument("--tokenizer", type=str, required=True,
                        help="HF hub id or local path.")
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--max-tokens", type=int, default=600)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(args.tokenizer))
    model = AutoModelForCausalLM.from_pretrained(str(args.source_model)).eval()
    mlp = model.model.layers[args.layer].mlp

    captured: list[np.ndarray] = []

    def pre_hook(_module, inputs):
        captured.append(inputs[0].detach().reshape(-1, inputs[0].shape[-1]).float().cpu().numpy())

    handle = mlp.register_forward_pre_hook(pre_hook)
    with torch.no_grad():
        for text in CALIBRATION:
            batch = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
            model(**batch)
    handle.remove()

    hidden = np.concatenate(captured, axis=0)
    if hidden.shape[0] > args.max_tokens:
        hidden = hidden[: args.max_tokens]

    gate = mlp.gate_proj.weight.detach().float().cpu().numpy()  # [I, H]
    up = mlp.up_proj.weight.detach().float().cpu().numpy()      # [I, H]
    down = mlp.down_proj.weight.detach().float().cpu().numpy()  # [H, I]

    from moeforge.grouping import channel_importance, intermediate_activations

    activations = intermediate_activations(hidden, gate, up).astype(np.float32)
    importance = channel_importance(activations).astype(np.float32)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        activations=activations,
        down=down.astype(np.float32),
        importance=importance,
        layer=args.layer,
        # Router-search tensors (examples/router-search): the FFN input hidden states a real
        # router sees, plus the gate/up rows it derives per-expert keys from. Extra keys are
        # ignored by grouping-search consumers.
        hidden=hidden.astype(np.float32),
        gate=gate.astype(np.float32),
        up=up.astype(np.float32),
    )
    print(
        f"wrote {args.output}  tokens={activations.shape[0]} channels={activations.shape[1]} "
        f"hidden={down.shape[0]}"
    )


if __name__ == "__main__":
    main()
