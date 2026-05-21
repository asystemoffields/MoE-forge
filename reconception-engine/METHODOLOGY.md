# Reconception Engine — methodology

A method for using an LLM-driven evolutionary loop to **reconceive a problem**, not just
optimize within a fixed framing. The ordinary loop (model proposes a *solution*, a metric
disposes) hill-climbs inside the representation you chose — it converges, it does not innovate.
This escalates it: the candidate is the **method / problem-formulation itself**, and an
invariant verifier judges only the true outcome, so the loop is free to reframe what the
problem *is* while staying honest.

## The core move: separate the mutable frame from the immutable judge
A problem formulation is the tuple **⟨objective, representation, constraints, cost-model,
abstraction-level⟩**. Every famous reconception is a transformation of that tuple:
- FlashAttention = swap the *cost model* (count memory movement, not FLOPs) → tiling falls out.
- TurboQuant = change *representation* (rotate to an incoherent basis) + *objective* (inner-product distortion, not MSE).
- AlphaTensor = *lift* the abstraction (matmul → tensor decomposition).

So "reconceiving" is not magic — it is mutating that tuple. Make the tuple a **program the LLM
mutates**; make the **judge external and invariant**. The judge measures the one thing no
framing may redefine: the true, real, hermetic outcome on held-out data. That invariance is
what keeps "creativity" from becoming self-delusion (every reframing looks good by its own lights).

## Architecture (bilevel)
- **L0 — anchor (immutable):** the true resource + true quality, as close to downstream reality
  as affordable, hermetic and held-out. **Its fidelity caps everything above it.**
- **L1 — formulation/method search (the creative engine):** candidates are *methods* (code),
  free to reconceive. Proposed by LLMs.
- **L2 — inner solve:** a cheap general solver (or the method runs itself) → a solution →
  decoded back to native space → scored by L0.

Win condition = a reframing whose *solution*, measured by L0, beats the incumbent frame's. The
reconception is "real" iff it moves the needle L0 can't fake.

## Make it explore, not hill-climb
- **Quality-Diversity (MAP-Elites):** archive cells = (resource bin) × (method-family). Fitness
  in a cell = quality. The archive *is* the rate-distortion frontier × method-space; a new
  family winning a cell = a reconception. Diversity (not single-best) surfaces the stepping
  stones that pure fitness search walks past.
- **Islands = stances:** spawn independent LLM proposers, each holding a different *conception*
  of the problem (e.g. info-theory / behavioral / representation / structure / "ignore
  convention"). Structurally different framings, in parallel.
- **Key-free generation:** no API key needed — Claude Code spawns Sonnet subagents as the
  generators; the local verifier scores them. Generation and evaluation stay separate (the
  model proposes, the verifier disposes).

## The verifier is the whole game (anti-Goodhart)
Open-ended search reward-hacks ferociously. The verifier MUST:
- count **everything that ships** (decoder code + state + payload) — the Hutter-prize rule. In
  the compression contract, `decompress` sees ONLY the artifact, so any data a method keeps
  costs bytes and cannot be smuggled into the decoder for free;
- measure quality through the **real** decode/forward path, on a **held-out** set disjoint from
  whatever the method is allowed to study;
- be **hermetic** (no network/disk/globals), deterministic, and enforce a **compute cap**
  (efficiency is part of the contract — a method that can't run in budget legitimately loses);
- [hardening] re-run for determinism; optionally run on a second model to kill hardcoded constants.

If L0 leaks or proxies, the loop "reconceives" straight into a cheat. Spend your design effort here.

## Validate against the landscape (after L0 says it's real)
Invert the usual order — **execution discovers, then you validate**:
1. hermetic re-score (held-out, true-resource) → kill artifacts;
2. **interp** explains *why* it works (the unique edge: you can open the black box on your own discovery);
3. **research-agents** situate it vs prior art (validators, not ideators — checking a concrete
   artifact against literature, not generating consensus);
4. stress-test across sizes / models / tasks.

Caveat: most "discoveries" are **rediscoveries** (novel-to-the-loop ≠ novel-to-the-world). Only
the literature check tells you which bucket — and a rediscovery is still open-re-derivation value.

## The honest ceiling
The loop reconceives the *formulation*, but it is anchored to whatever you define as true-good
(L0). It can discover that the binding constraint is IO not FLOPs — because wall-clock (L0)
reveals it. It **cannot** discover it should care about something you never measured. So the
human move lifts from "reframe the problem" to **"define what ultimately counts."** That regress
bottoms out at contact with reality: the only ungameable anchor is real downstream utility. Make
L0 as close to that as you can afford.

## Recipe (apply to a new problem)
1. **Write L0** — the true, hermetic, held-out outcome metric. Hardest and most important step.
2. **Define the candidate contract** as a *method* (code) with maximal freedom over *how* it
   achieves the outcome (it may reconceive what to preserve / optimize).
3. **Seed** 2–3 baselines, including a trivial/lossless corner, to anchor the frontier.
4. **Pick behavioral descriptors** → MAP-Elites cells (resource bin × method-family).
5. **Write the reconception prompt:** the contract + explicit freedom to redefine what's
   preserved + a move vocabulary (relaxation, change-of-basis, cost-model substitution,
   constraint drop, lifting, entropy coding, …) + the rules L0 enforces.
6. **Spawn stance-subagents** (islands) → candidates → score with L0 → insert into archive → repeat.
7. **Validate winners** against the landscape.

## Status / worked example: LLM compression (`examples/compress-evolve/`)
- **L0 verifier built and validated** on SmolLM-135M (ships bf16, so ratios are vs the
  2-byte/param model): `noop` = 1.00× / +0.000 NLL (lossless corner); `int8 RTN` = 1.65× /
  +0.096. Runs in ~tens of seconds on CPU → trustable on the unknown (the judge catches garbage,
  confirms gold).
- **Gen-1 (4 stances, key-free Sonnet subagents):** all four FAILED, and the verifier caught
  every failure mode distinctly — behavioral → NaN weights; representation → NLL ~2e5 (garbage);
  structure → OOM; info → timeout (over compute budget). None beat int8. This is the expected,
  useful gen-1 signal: the machine and the verifier work (it is unfoolable — 2.4× "compression"
  scores nothing when NLL is NaN), and the candidates need iteration. Gen-2 feedback: vectorize
  (no per-element Python loops), bound memory, stay conservative enough that reconstruction is finite.
- **What's still to build:** the formal MAP-Elites archive (`tools/sonnet-evolve/qd.py`) and a
  generation driver; gen-1 was driven manually. The frontier today = the seeds (`noop`, `int8_rtn`).

## Lineage
This is the escalation of the DIY-AlphaEvolve thesis — **optimizer → cartographer (QD) →
reframer (reconception)** — on the same harness (`tools/sonnet-evolve/`). Earlier single-objective
instances: `examples/grouping-search/` (carve channel grouping), `examples/router-search/`
(top-k router selection). The leverage is in the verifier (L0) and the diversity engine, not in
any single candidate.
