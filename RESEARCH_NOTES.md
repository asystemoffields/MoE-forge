# Research Notes

## Closing the carve 90->95 retention gap (2026-05-21/22) — three structural levers, all negative

The carved MoE (SmolLM-135M, 4 experts) recovers to ~90% of dense at top-3/4 (mild sparsity); the
carve is near-lossless, so the gap is the sparse top-k routing. Three attempts to close the last ~5pts:

1. **Post-hoc low-rank residual** (fit a rank-r linear map to correct `forward_all - forward_token_router`):
   captures only ~6% of the per-layer error (held-out rel-err 0.94), end-to-end NLL **+0.165 (worse)**.
   The dropped-expert error is nonlinear + routing-dependent, not low-rank-linear in the layer input.
   (`examples/residual-search/`)

2. **Adaptive per-layer k** (redistribute a fixed avg-k budget): NLL **+0.32 (worse)** at matched active.
   The expert-budget curve is a cliff (uniform top2=5.72, top3=3.85, top4=3.87 no-gain): 3 experts are
   needed everywhere and the 4th is dead, so uniform top-3 is already optimal — no allocation headroom;
   E=4 is too coarse. (`examples/adaptive-k/`)

3. **Finer granularity (8 experts, top-6/8, matched active fraction)**: **84.9% avg retention — ~5pts
   WORSE than the 4-expert top-3's 90.0%.** An oracle-reconstruction gate showed the 8-expert carve has a
   ~6% lower recon FLOOR (30/30 layers), but the LEARNED 8-way top-6 router is harder to recover than
   4-way top-3, so the deployed routing ate the floor gain (oracle selection != learned router — the gate
   measured the wrong thing). Recovered teacher-KL 0.225 vs the 4-expert's 0.199.
   (`examples/grouping-search/{carve_evolved_8,gate_oracle_recon}.py`) Also: `evolved_refine` "better
   grouping" collapses to ~4 big experts under the 2x balance cap (coverage-optimization fights finer
   granularity), so finer+better-grouping needs a tight-evenness constraint refine doesn't expose.

**Conclusion:** ~90% is carve's practical ceiling at this scale/recipe. Recon/KL gains repeatedly fail to
translate to retention (the floor improves but the learned router can't realize it). The mission-aligned
path is **carve-at-mild-sparsity (~90%) x PMRA quantization**, not forcing carve alone to 95%. Levers not
yet exhausted (uncertain): far larger recovery data/steps; a *trained-in* (not post-hoc) shared/low-rank
capacity; a better learned-router init for higher expert counts; or upcycle instead of carve.

## EMO And Document-Level Modularity

Ai2's EMO work is directly relevant to MoE Forge's router and profiling roadmap:

- Blog: https://allenai.org/blog/emo
- Paper: https://arxiv.org/abs/2605.06663
- Code: https://github.com/allenai/EMO

EMO's key idea is to encourage higher-level modularity by constraining tokens from the same document to route within a shared expert pool. The pool is selected from router preferences averaged over document tokens, while global load balancing keeps expert usage healthy across many documents. Ai2 reports that EMO keeps near full-model performance with selective expert subsets, unlike a matched standard MoE.

MoE Forge should adapt this as a dense-to-MoE conversion principle:

1. Preserve calibration sample identity during profiling. Initial support exists through per-document profile summaries keyed by stable text hashes.
2. Collect per-document FFN channel summaries alongside global channel summaries.
3. Compare global-importance carving against document-cluster carving.
4. Add router metadata for a `document_pool_then_token_router` strategy. Initial profiling reports and router-plan artifacts now include first-pass document expert-pool recommendations.
5. Evaluate selected expert subsets with `keep_k` sweeps.

Near-term experiment:

```text
profile calibration documents
build per-document channel vectors
cluster documents by channel usage
carve shared channels from global importance
carve routed experts from document-cluster-specific channels
compare against greedy/global and random/balanced baselines
```

Evaluation should include:

- dense baseline vs full carved MoE
- full carved MoE vs selected expert subsets
- teacher KL/perplexity by document group
- active experts per document
- memory/quality curves as expert pool size changes

## Carve Grouping & Sparsity Findings (2026-05)

Measured on SmolLM-135M FFN layers via oracle-top-k reconstruction error (see
`examples/grouping-search/`). These guide carve defaults and strategy selection.

- **Sparsity has no free lunch.** Un-recovered reconstruction error scales ~linearly with the
  active fraction (active channels per token). There is no sparsity "knee" — every bit of
  sparsity costs proportional quality. Recovery training is what bends this curve.
- **Recovery training bends it but is finite.** Joint expert+router recovery took a sparse
  carve to ~89% benchmark retention (top-3/4), but teacher-KL gains translate weakly to
  retention (a large KL drop bought ~1pt). Recovery is largely tapped near ~89%.
- **Grouping matters, modestly.** At fixed (balanced) sparsity, clustering routed channels by
  their **absolute-value** activation vectors (co-firing under the gated FFN, opposite-sign
  partners grouped together) or squared activations beats the magnitude/importance-balance
  default by ~4-5% on held-out layers. See `balanced_grouping`.
- **Fine-grained experts help, at fixed compute.** Holding the active fraction constant, more
  (smaller) experts with proportionally higher top-k lowers reconstruction error ~6-11%
  (8→48 experts). For carve this is ~free (same channels, finer partition). Caveat: a *learned*
  router over many experts may not realize the full *oracle* gain.
- **These stack.** Grouping + granularity together cut reconstruction ~17% at the same active
  fraction (0.528→0.436), un-recovered.
- **Overlap does NOT help.** A carve↔upcycle hybrid (a channel duplicated into several experts)
  is worse than disjoint experts at matched active compute, and costs 1.9-2.8x memory. The
  oracle already selects the best top-k; duplicating channels makes experts fat and redundant
  rather than giving the router better choices. Granularity (finer disjoint) is the lever, not
  overlap.

**Implication for goldilocks ("sparse + near-dense"):** chase it via stacked levers
(fine-grained + abs grouping + stronger shared + recovery) at *mild* sparsity, not aggressive
sparsity. Carve's edge is small-model + low-compute; pair it with quantization (PMRA) for the
local-deployment win.

## Strategy Selection

`moe-forge plan --strategy {carved_mlp|sparse_upcycle|adapter_moe}` (auto from `--goal` if
unset). Tradeoffs the planner records:

- **carved_mlp** — partition the dense FFN into shared + routed experts. ~dense params (smaller,
  quantizable), sparse compute. Quality is the hard part (see findings above). The only backend
  built end-to-end today.
- **sparse_upcycle** — replicate the FFN into N full-width experts + router, then train. More
  params (bigger memory) but reliably reaches high quality; the proven path. Construction
  backend planned.
- **adapter_moe** — LoRA/adapter experts on a frozen dense trunk. Cheap, good for laptop
  experiments and domain specialization. Construction backend planned.
