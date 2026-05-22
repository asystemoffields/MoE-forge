"""Build a custom carve PROFILE for an 8-expert carve of SmolLM-135M using the evolved_refine
grouping (the best generalizer from grouping-search), to test whether finer granularity + better
grouping closes the 90->95 retention gap.

For every FFN layer: capture the MLP input hidden states on calib, compute gated activations and
per-channel importance, run evolved_refine.group(...) at n_experts experts, and emit the carve's
first-class profile JSON (target.layer/role + assignment with shared_channels + per-expert
channels). Per layer falls back to the magnitude grouping if the evolved grouping errors, so the
carve always has a valid assignment.

Output feeds: moe-forge carve-manifest <model> --recipe recipe.json --profile <this> --output ...

Usage:
  python examples/grouping-search/carve_evolved_8.py --output outputs/residual-search/profile8.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "examples" / "grouping-search" / "candidates"))

from moeforge.grouping import channel_importance, intermediate_activations, magnitude_grouping  # noqa: E402

try:
    import evolved_refine  # noqa: E402
    _HAVE_REFINE = True
except Exception as exc:  # pragma: no cover
    print(f"WARNING: could not import evolved_refine ({exc}); using magnitude for all layers")
    _HAVE_REFINE = False


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source-model", default="HuggingFaceTB/SmolLM-135M")
    ap.add_argument("--experts", type=int, default=8)
    ap.add_argument("--shared-ratio", type=float, default=256.0 / 1536.0)
    ap.add_argument("--calib-file", default="outputs/residual-search/eval.txt")
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--max-tokens", type=int, default=1280)
    ap.add_argument("--time-limit", type=float, default=12.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.source_model)
    model = AutoModelForCausalLM.from_pretrained(args.source_model).eval()
    layers = model.model.layers
    n_layers = len(layers)
    print(f"model={args.source_model} layers={n_layers} experts={args.experts} "
          f"shared_ratio={args.shared_ratio:.4f}")

    captured = {i: [] for i in range(n_layers)}
    handles = []
    for i, lyr in enumerate(layers):
        def pre(_m, inp, _i=i):
            captured[_i].append(inp[0].detach().reshape(-1, inp[0].shape[-1]).float())
        handles.append(lyr.mlp.register_forward_pre_hook(pre))

    text = Path(args.calib_file).read_text(encoding="utf-8", errors="replace")
    ids = tokenizer(text, return_tensors="pt").input_ids[0]
    n_seq = min(args.max_tokens // args.seq_len, ids.shape[0] // args.seq_len)
    batches = ids[:n_seq * args.seq_len].reshape(-1, args.seq_len)
    print(f"calib: {n_seq} seqs x {args.seq_len} = {n_seq * args.seq_len} tokens")
    with torch.no_grad():
        for b in range(batches.shape[0]):
            model(batches[b:b + 1])
    for h in handles:
        h.remove()

    profile = {"modules": {}}
    for i in range(n_layers):
        hidden = torch.cat(captured[i], 0).numpy().astype(np.float64)
        gate = layers[i].mlp.gate_proj.weight.detach().float().cpu().numpy()
        up = layers[i].mlp.up_proj.weight.detach().float().cpu().numpy()
        acts = intermediate_activations(hidden, gate, up).astype(np.float64)
        imp = channel_importance(acts).astype(np.float64)

        method = "evolved_refine"
        assign = None
        if _HAVE_REFINE:
            try:
                rng = np.random.default_rng(args.seed)
                ctx = {"importance": imp, "activations": acts}
                assign = np.asarray(evolved_refine.group(
                    ctx, args.experts, args.shared_ratio, rng, time_limit=args.time_limit))
                # sanity: right shape + all experts used + valid labels
                ok = (assign.shape[0] == imp.shape[0]
                      and set(np.unique(assign[assign >= 0]).tolist()) == set(range(args.experts)))
                if not ok:
                    raise ValueError(f"invalid assignment (labels={np.unique(assign).tolist()})")
            except Exception as exc:
                method = f"magnitude_fallback({type(exc).__name__})"
                assign = None
        if assign is None:
            if method == "evolved_refine":
                method = "magnitude"
            assign = magnitude_grouping(imp, n_experts=args.experts, shared_ratio=args.shared_ratio)

        width = int(assign.shape[0])
        shared = [int(c) for c in np.where(assign == -1)[0].tolist()]
        experts = [{"expert": e, "channels": [int(c) for c in np.where(assign == e)[0].tolist()]}
                   for e in range(args.experts)]
        profile["modules"][f"model.layers.{i}.mlp.gate_proj"] = {
            "target": {"layer": i, "role": "gate"},
            "assignment": {"available": True, "width": width,
                           "shared_channels": shared, "experts": experts},
        }
        sizes = [len(e["channels"]) for e in experts]
        print(f"layer {i:2d} [{method:24}] shared={len(shared):4d} experts={sizes}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(profile))
    n_refine = sum(1 for m in profile["modules"].values() if m)  # all written
    print(f"wrote {args.output}  ({n_layers} layers, {args.experts} experts)")


if __name__ == "__main__":
    main()
