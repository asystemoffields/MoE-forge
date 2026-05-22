"""Prototype: fit an always-on low-rank RESIDUAL that corrects the carved-MoE top-k routing
error, and measure the end-to-end NLL change. Standalone (cf. grouping-search / router-search).

The carve is ~lossless with ALL experts; the entire quality gap is top-k routing dropping
experts. Per MoE layer, on the activations the SPARSE model actually sees:

    e = forward_all(x) - forward_token_router(x)        # the dropped-expert error (per token)

We fit a per-layer rank-r map R (R x ~= e) by ridge regression + truncated SVD, attach R(x)
always-on at the end of forward_token_router, and compare held-out NLL with vs without R.
The with/without delta is measured on the SAME held-out text, so it's robust to held-out size.

Caveat: residuals are fit per-layer on no-R activations then stacked (attaching R_L shifts the
input to L+1). Fine for a first read; a sequential refit would only help.

Usage:
  python examples/residual-search/fit_residual.py --wrapper outputs/residual-search/top3-wrapper --rank 32
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from moeforge.hf_runtime import MoEForgeCarvedMLPModule  # noqa: E402

# Diverse general text: prose, science, code, reasoning. Split into fit / held-out below.
CORPUS = [
    "The printing press, introduced in fifteenth-century Europe, made the mass reproduction of text possible and accelerated the spread of literacy and ideas across the continent.",
    "A binary search repeatedly halves the interval that could contain the target, so it finds an element among a million sorted values in about twenty comparisons.",
    "Photosynthesis converts carbon dioxide and water into glucose and oxygen using energy from sunlight captured by chlorophyll in the chloroplasts of plant cells.",
    "When a central bank raises its policy interest rate, borrowing becomes more expensive, demand cools, and inflation tends to ease over the following quarters.",
    "def quicksort(xs):\n    if len(xs) <= 1:\n        return xs\n    pivot = xs[len(xs) // 2]\n    left = [x for x in xs if x < pivot]\n    mid = [x for x in xs if x == pivot]\n    right = [x for x in xs if x > pivot]\n    return quicksort(left) + mid + quicksort(right)",
    "Tectonic plates drift only a few centimeters each year, yet over millions of years that motion opens oceans, raises mountain ranges, and rearranges entire coastlines.",
    "She turned the dial of the old radio slowly until, through the hiss of static, a faint orchestral melody steadied into something she could almost hum along to.",
    "In linear algebra the rank of a matrix equals the dimension of its column space, which is also the number of nonzero singular values it has.",
    "The adaptive immune system keeps memory cells after an infection, so a second encounter with the same pathogen is recognized and cleared far more quickly than the first.",
    "A cache trades memory for speed by keeping recently or frequently used results close to the processor, so repeated work can skip the slow path entirely.",
    "The 1648 treaty redrew the contested border along the course of the river, ending nearly a decade of intermittent fighting between the two exhausted kingdoms.",
    "Enzymes are biological catalysts that lower the activation energy of a reaction, letting it proceed quickly at body temperature without being consumed in the process.",
    "import numpy as np\n\ndef softmax(z):\n    z = z - z.max(axis=-1, keepdims=True)\n    e = np.exp(z)\n    return e / e.sum(axis=-1, keepdims=True)",
    "Convection carries heat upward as warm, less dense fluid rises and cooler fluid sinks to take its place, a cycle that drives weather, ocean currents, and the boiling of a pot of water.",
    "The novel's narrator is unreliable: small contradictions in his account accumulate until the reader realizes the comforting story he tells is one he needs to believe.",
    "To estimate the height of a tree without climbing it, measure the length of its shadow and compare it to the shadow of a stick of known height at the same time of day.",
    "A compiler translates source code into machine instructions in stages: lexing into tokens, parsing into a syntax tree, checking types, and finally emitting optimized code.",
    "Coral reefs support a quarter of all marine species despite covering a tiny fraction of the ocean floor, which is why even small rises in water temperature are so dangerous.",
    "The committee debated for hours, but the decisive argument was simple: the cheaper plan would cost far more later, once the deferred repairs finally came due.",
    "In probability, two events are independent when knowing that one occurred does not change the likelihood of the other, so their joint probability is just the product of the two.",
    "Sailing against the wind is possible by tacking: the sail acts like a wing, and the keel resists sideways motion, so the boat advances in a zigzag toward the wind.",
    "The library at Alexandria aimed to collect every scroll in the known world, and ships entering the harbor were searched so their books could be copied for the shelves.",
    "Gradient descent nudges each parameter a small step in the direction that most reduces the loss, and with a well-chosen step size it converges toward a local minimum.",
    "After the rain the desert bloomed within days: seeds that had waited years for moisture raced to flower and set new seed before the brief wet window closed again.",
]


def make_batches(tokenizer, text, seq_len, max_seqs):
    ids = tokenizer(text, return_tensors="pt").input_ids[0]
    usable = (ids.shape[0] // seq_len) * seq_len
    batches = ids[:usable].reshape(-1, seq_len)
    return batches[:max_seqs]


def seq_nll(model, batches):
    total, count = 0.0, 0
    with torch.no_grad():
        for i in range(batches.shape[0]):
            ids = batches[i:i + 1]
            out = model(ids, labels=ids)
            total += float(out.loss)
            count += 1
    return total / max(count, 1)


def moe_layers(model):
    return [(n, m) for n, m in model.named_modules() if isinstance(m, MoEForgeCarvedMLPModule)]


def capture_inputs(model, mods, batches):
    """One forward pass; pre-hooks grab each MoE module's input x as [N, H] (fp32)."""
    store = {n: [] for n, _ in mods}
    handles = []
    for n, m in mods:
        def pre(_mod, inp, _n=n):
            x = inp[0].detach().reshape(-1, inp[0].shape[-1]).float()
            store[_n].append(x)
        handles.append(m.register_forward_pre_hook(pre))
    with torch.no_grad():
        for i in range(batches.shape[0]):
            model(batches[i:i + 1])
    for h in handles:
        h.remove()
    return {n: torch.cat(v, 0) for n, v in store.items()}


def _module_dtype(module):
    for b in module.buffers():
        if b.is_floating_point():
            return b.dtype
    return torch.float32


def residual_targets(module, x):
    """e = forward_all(x) - forward_token_router(x), in fp32, on the captured activations."""
    xc = x.to(_module_dtype(module))
    with torch.no_grad():
        y_all = module.forward_all(xc).float()
        y_topk = module.forward_token_router(xc).float()
    return y_all - y_topk


def fit_lowrank(x, e, rank, lam):
    """Ridge-regress e on x (e ~= x @ W), then truncate W to `rank` via SVD.
    Returns A1 [H, r], A2 [r, H] with R(x) = (x @ A1) @ A2."""
    H = x.shape[1]
    XtX = x.T @ x
    XtE = x.T @ e
    W = torch.linalg.solve(XtX + lam * torch.eye(H, dtype=x.dtype), XtE)  # [H, H], e ~= x @ W
    U, S, Vh = torch.linalg.svd(W, full_matrices=False)
    r = min(rank, S.shape[0])
    A1 = U[:, :r].contiguous()                       # [H, r]
    A2 = (S[:r].unsqueeze(1) * Vh[:r, :]).contiguous()  # [r, H]
    return A1, A2


def rel_error(x, e, A1, A2):
    """||e - R(x)|| / ||e|| on held-out activations."""
    pred = (x @ A1) @ A2
    return float(torch.linalg.norm(e - pred) / (torch.linalg.norm(e) + 1e-9))


def attach_residual(module, A1, A2):
    module._res_A1 = A1
    module._res_A2 = A2
    orig = module.forward_token_router

    def patched(hidden, _orig=orig, _m=module):
        out = _orig(hidden)
        a1 = _m._res_A1.to(hidden.dtype)
        a2 = _m._res_A2.to(hidden.dtype)
        return out + (hidden @ a1) @ a2

    module.forward_token_router = patched


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--wrapper", type=Path, required=True)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--lam", type=float, default=1e-2)
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--fit-seqs", type=int, default=40)
    ap.add_argument("--eval-seqs", type=int, default=16)
    ap.add_argument("--corpus-file", default=None, help="Local text file for calib/held-out (else inline CORPUS).")
    args = ap.parse_args()

    torch.manual_seed(0)
    print(f"loading wrapper: {args.wrapper}")
    tokenizer = AutoTokenizer.from_pretrained(str(args.wrapper))
    model = AutoModelForCausalLM.from_pretrained(
        str(args.wrapper), trust_remote_code=True, moeforge_expert_mode="learned-router"
    ).eval()

    mods = moe_layers(model)
    print(f"carved MoE layers found: {len(mods)}")

    if args.corpus_file:
        corpus_text = Path(args.corpus_file).read_text(encoding="utf-8", errors="replace")[:200000]
    else:
        corpus_text = "\n\n".join(CORPUS)
    batches = make_batches(tokenizer, corpus_text, args.seq_len, args.fit_seqs + args.eval_seqs)
    fit_b = batches[:args.fit_seqs]
    eval_b = batches[args.fit_seqs:args.fit_seqs + args.eval_seqs]
    print(f"fit seqs={fit_b.shape[0]} eval seqs={eval_b.shape[0]} (seq_len={args.seq_len})")

    nll_before = seq_nll(model, eval_b)
    print(f"NLL (top-k, no residual): {nll_before:.4f}")

    print("capturing fit activations ...")
    x_fit = capture_inputs(model, mods, fit_b)
    print("capturing eval activations ...")
    x_eval = capture_inputs(model, mods, eval_b)

    fits = {}
    rels = []
    for n, m in mods:
        xf = x_fit[n]
        ef = residual_targets(m, xf)
        A1, A2 = fit_lowrank(xf, ef, args.rank, args.lam)
        xe = x_eval[n]
        ee = residual_targets(m, xe)
        rel = rel_error(xe, ee, A1, A2)
        rels.append(rel)
        fits[n] = (A1, A2)
        print(f"  {n:40} rel-err(e->R) held-out: {rel:.3f}")

    print(f"mean held-out residual rel-err: {np.mean(rels):.3f} (rank={args.rank}, lam={args.lam})")

    for n, m in mods:
        A1, A2 = fits[n]
        attach_residual(m, A1, A2)

    nll_after = seq_nll(model, eval_b)
    print(f"NLL (top-k + residual):   {nll_after:.4f}")
    print(f"NLL delta (after-before): {nll_after - nll_before:+.4f}  (negative = residual helps)")


if __name__ == "__main__":
    main()
