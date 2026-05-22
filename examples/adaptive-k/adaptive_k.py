"""Adaptive per-layer expert budget (adaptive-k) for a carved-MoE wrapper.

Uniform top-k spends the same number of experts at every layer, but the sparse-routing loss is
not uniform across layers. This allocates a FIXED average-k budget (matched to uniform top-3 =
same active fraction) NON-uniformly: layers whose output degrades most when starved get more
experts, flat layers get fewer. Measured entirely within the deployed router-weighted scheme
(forward_token_router), so k = expert_count (router-weighted all) is the reference.

Per layer, on calib activations x captured at uniform top-3:
    y_ref    = forward_token_router(x; k=E)                  # best the router scheme achieves
    err_L(k) = || forward_token_router(x; k) - y_ref ||^2    (k = 2 .. E-1; err_L(E)=0)
Greedily spend the extra-expert budget on the largest marginal error reductions, set each
layer's token_router_top_k, and compare end-to-end NLL to uniform top-3 (same total budget).

Caveat: the router+experts were trained for uniform top-3; pushing a layer to k=2 at deploy
makes it sparser than it was trained for (a handicap on adaptive-k). If adaptive still wins,
that's a strong read; a fair ceiling would retrain at the allocation.

Usage:
  python examples/adaptive-k/adaptive_k.py --wrapper outputs/residual-search/w3 \
      --corpus-file outputs/residual-search/eval.txt
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from moeforge.hf_runtime import MoEForgeCarvedMLPModule  # noqa: E402


def make_batches(tokenizer, text, seq_len, max_seqs):
    ids = tokenizer(text, return_tensors="pt").input_ids[0]
    usable = (ids.shape[0] // seq_len) * seq_len
    return ids[:usable].reshape(-1, seq_len)[:max_seqs]


def seq_nll(model, batches):
    total, count = 0.0, 0
    with torch.no_grad():
        for i in range(batches.shape[0]):
            ids = batches[i:i + 1]
            total += float(model(ids, labels=ids).loss)
            count += 1
    return total / max(count, 1)


def moe_layers(model):
    return [(n, m) for n, m in model.named_modules() if isinstance(m, MoEForgeCarvedMLPModule)]


def module_dtype(m):
    for b in m.buffers():
        if b.is_floating_point():
            return b.dtype
    return torch.float32


def capture_inputs(model, mods, batches):
    store = {n: [] for n, _ in mods}
    handles = []
    for n, m in mods:
        def pre(_mod, inp, _n=n):
            store[_n].append(inp[0].detach().reshape(-1, inp[0].shape[-1]))
        handles.append(m.register_forward_pre_hook(pre))
    with torch.no_grad():
        for i in range(batches.shape[0]):
            model(batches[i:i + 1])
    for h in handles:
        h.remove()
    return {n: torch.cat(v, 0) for n, v in store.items()}


def set_uniform_k(mods, k):
    for _, m in mods:
        m.token_router_top_k = int(k)


def layer_id(name):
    parts = name.split(".")
    return parts[-2] if len(parts) >= 2 else name


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--wrapper", type=Path, required=True)
    ap.add_argument("--corpus-file", required=True)
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--measure-seqs", type=int, default=20)
    ap.add_argument("--eval-seqs", type=int, default=20)
    ap.add_argument("--avg-k", type=int, default=3, help="matched uniform-k budget baseline")
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(str(args.wrapper))
    model = AutoModelForCausalLM.from_pretrained(
        str(args.wrapper), trust_remote_code=True, moeforge_expert_mode="learned-router"
    ).eval()
    mods = moe_layers(model)
    E = int(mods[0][1].expert_count)
    n = len(mods)
    print(f"layers={n} expert_count={E}")

    text = Path(args.corpus_file).read_text(encoding="utf-8", errors="replace")[:200000]
    batches = make_batches(tokenizer, text, args.seq_len, args.measure_seqs + args.eval_seqs)
    meas_b = batches[:args.measure_seqs]
    eval_b = batches[args.measure_seqs:args.measure_seqs + args.eval_seqs]
    print(f"measure seqs={meas_b.shape[0]} eval seqs={eval_b.shape[0]}")

    # reference end-to-end NLLs at uniform k
    refs = {}
    for k in sorted({2, args.avg_k, E}):
        set_uniform_k(mods, k)
        refs[k] = seq_nll(model, eval_b)
    print("uniform NLL: " + "  ".join(f"top{k}={refs[k]:.4f}" for k in sorted(refs)))

    # capture per-layer x at uniform top-(avg_k) routing
    set_uniform_k(mods, args.avg_k)
    x_by = capture_inputs(model, mods, meas_b)

    # per-layer err(k) vs k=E, within the router-weighted scheme
    errs = {}
    for name, m in mods:
        x = x_by[name].to(module_dtype(m))
        with torch.no_grad():
            m.token_router_top_k = E
            y_ref = m.forward_token_router(x).float()
            d = {E: 0.0}
            for k in range(2, E):
                m.token_router_top_k = k
                yk = m.forward_token_router(x).float()
                d[k] = float(((yk - y_ref) ** 2).sum())
        m.token_router_top_k = args.avg_k
        errs[name] = d

    # greedy allocation: start all at k=2, spend (budget - 2n) extra experts on max marginal gains
    budget = n * args.avg_k
    kalloc = {name: 2 for name, _ in mods}
    for _ in range(budget - 2 * n):
        best, best_gain = None, -1.0
        for name, _ in mods:
            k = kalloc[name]
            if k >= E:
                continue
            gain = errs[name][k] - errs[name][k + 1]
            if gain > best_gain:
                best_gain, best = gain, name
        if best is None:
            break
        kalloc[best] += 1

    for name, m in mods:
        m.token_router_top_k = kalloc[name]
    nll_adaptive = seq_nll(model, eval_b)

    hist = Counter(kalloc.values())
    print(f"\nadaptive allocation (sum={sum(kalloc.values())}, budget={budget}): "
          + "  ".join(f"k={k}:{hist[k]}L" for k in sorted(hist)))
    print(f"  k={E} layers: {[layer_id(n) for n, _ in mods if kalloc[n] == E]}")
    print(f"  k=2 layers: {[layer_id(n) for n, _ in mods if kalloc[n] == 2]}")
    base = refs[args.avg_k]
    print(f"\nNLL  uniform top{args.avg_k}={base:.4f}   ADAPTIVE(avg{args.avg_k})={nll_adaptive:.4f}"
          f"   delta={nll_adaptive - base:+.4f}  (negative = adaptive wins)")
    print(f"  reference range: top2={refs[2]:.4f} ... top{E}(all)={refs[E]:.4f}")


if __name__ == "__main__":
    main()
