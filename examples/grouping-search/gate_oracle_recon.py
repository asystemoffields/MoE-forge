"""Gate + profile for the finer-granularity carve.

Compares oracle top-k reconstruction error of an 8-expert MAGNITUDE carve at top-6 (matched active
fraction) vs the 4-expert magnitude baseline at top-3, per FFN layer, and writes the 8-expert
magnitude profile for carving. oracle_topk_error picks the best k experts per token (no trained
router), so it measures the carve's representational FLOOR. If finer (8@top6) beats the baseline
(4@top3) here, the floor is lower and the Modal recovery is worth running.

(Note: the evolved_refine grouping collapsed to ~4 big experts under the 2x balance cap -- a
coverage-optimized grouping concentrates channels, defeating finer granularity -- so this uses the
even magnitude grouping. Combining "finer" with "better grouping" would need a tight evenness
constraint that evolved_refine does not expose.)

Usage:
  python examples/grouping-search/gate_oracle_recon.py --profile-out outputs/residual-search/profile8_mag.json
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
from moeforge.grouping import (  # noqa: E402
    channel_importance, intermediate_activations, magnitude_grouping, oracle_topk_error,
)


def profile_entry(assign, layer, n_experts):
    width = int(assign.shape[0])
    shared = [int(c) for c in np.where(assign == -1)[0].tolist()]
    experts = [{"expert": e, "channels": [int(c) for c in np.where(assign == e)[0].tolist()]}
               for e in range(n_experts)]
    return {"target": {"layer": layer, "role": "gate"},
            "assignment": {"available": True, "width": width,
                           "shared_channels": shared, "experts": experts}}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source-model", default="HuggingFaceTB/SmolLM-135M")
    ap.add_argument("--shared-ratio", type=float, default=256.0 / 1536.0)
    ap.add_argument("--base-experts", type=int, default=4)
    ap.add_argument("--base-topk", type=int, default=3)
    ap.add_argument("--new-experts", type=int, default=8)
    ap.add_argument("--new-topk", type=int, default=6)
    ap.add_argument("--calib-file", default="outputs/residual-search/eval.txt")
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--max-tokens", type=int, default=1280)
    ap.add_argument("--profile-out", required=True)
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.source_model)
    model = AutoModelForCausalLM.from_pretrained(args.source_model).eval()
    layers = model.model.layers
    n_layers = len(layers)

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
    with torch.no_grad():
        for b in range(batches.shape[0]):
            model(batches[b:b + 1])
    for h in handles:
        h.remove()

    profile = {"modules": {}}
    base_errs, new_errs = [], []
    for i in range(n_layers):
        hidden = torch.cat(captured[i], 0).numpy().astype(np.float64)
        gate = layers[i].mlp.gate_proj.weight.detach().float().cpu().numpy()
        up = layers[i].mlp.up_proj.weight.detach().float().cpu().numpy()
        down = layers[i].mlp.down_proj.weight.detach().float().cpu().numpy()
        acts = intermediate_activations(hidden, gate, up).astype(np.float64)
        imp = channel_importance(acts).astype(np.float64)

        a_base = magnitude_grouping(imp, n_experts=args.base_experts, shared_ratio=args.shared_ratio)
        e_base = oracle_topk_error(activations=acts, down=down, assignment=a_base, top_k=args.base_topk)
        a_new = magnitude_grouping(imp, n_experts=args.new_experts, shared_ratio=args.shared_ratio)
        e_new = oracle_topk_error(activations=acts, down=down, assignment=a_new, top_k=args.new_topk)
        profile["modules"][f"model.layers.{i}.mlp.gate_proj"] = profile_entry(a_new, i, args.new_experts)
        base_errs.append(e_base)
        new_errs.append(e_new)
        print(f"layer {i:2d}  base({args.base_experts}@top{args.base_topk})={e_base:.4f}  "
              f"new({args.new_experts}@top{args.new_topk})={e_new:.4f}  "
              f"{'BETTER' if e_new < e_base else 'worse'} ({e_new - e_base:+.4f})")

    Path(args.profile_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.profile_out).write_text(json.dumps(profile))

    mb, mn = float(np.mean(base_errs)), float(np.mean(new_errs))
    wins = sum(1 for b, n in zip(base_errs, new_errs) if n < b)
    print(f"\nwrote {args.profile_out} (8-expert magnitude profile)")
    print(f"MEAN oracle recon  base({args.base_experts}@top{args.base_topk})={mb:.4f}   "
          f"new({args.new_experts}@top{args.new_topk})={mn:.4f}   delta={mn - mb:+.4f}  "
          f"({wins}/{n_layers} layers better)")
    verdict = "PASS" if mn < mb else "FAIL"
    print(f"GATE: {verdict} -- finer granularity {'lowers' if mn < mb else 'does NOT lower'} "
          f"the oracle floor at matched active fraction")


if __name__ == "__main__":
    main()
