# reconception-engine

Methodology for using an LLM-driven evolutionary loop to **reconceive a problem** — to evolve the
problem *formulation/method itself* against an invariant, hermetic verifier, rather than
hill-climbing a fixed metric in a representation you pre-chose. The goal is genuine novelty
(new methods, new framings), validated instantly because the verifier is cheap and unforgeable.

**Read [METHODOLOGY.md](METHODOLOGY.md)** for the full design: the core move (separate the
mutable frame from the immutable judge), the bilevel architecture (L0 anchor / L1 method search /
L2 inner solve), the exploration engine (quality-diversity + island "stances" + key-free Sonnet
subagent generation), the anti-Goodhart verifier discipline, the validate-against-the-landscape
loop, the honest ceiling, and a step-by-step recipe.

## Instances in this repo
- `examples/compress-evolve/` — the worked example: reconceive **LLM compression** (candidate =
  a complete `compress`/`decompress` method, free to preserve weights / activations / behavior /
  anything; judged only on held-out quality-per-shipped-byte). L0 verifier validated; gen-1 run done.
- `tools/sonnet-evolve/` — the base evolve loop the engine is built on.
- `examples/router-search/`, `examples/grouping-search/` — earlier single-objective instances.

## Status
L0 verifier validated on SmolLM-135M. Gen-1 (4 stances) ran end-to-end; all four failed and the
verifier caught each failure mode (NaN / garbage / OOM / timeout) — the judge is proven; the
candidates need iteration. Next: the formal MAP-Elites archive + generation driver, then gen-2.
