# GGUF Export Bridge — Scoping

Goal: export a MoE Forge model (carved **or** upcycled) to a **GGUF** file that llama.cpp runs,
so it can be (a) quantized by PMRA and (b) evaluated fast (vectorized MoE routing instead of our
Python per-token expert loop). This bridge unblocks **carve×PMRA** and **upcycle×PMRA**, and
fixes the slow-eval problem.

## Core strategy: target an architecture llama.cpp already runs — do NOT write C++

MoE Forge's structure — **shared expert (always on) + N routed experts + a learned softmax top-k
router + SwiGLU** — is the **DeepSeek-MoE** / **Qwen2-MoE** shape. llama.cpp already implements
both (`deepseek2`, `qwen2moe`). So the bridge is a **tensor-name + metadata mapping onto an
existing arch**, not a new runtime. **Recommended target: `deepseek2`** — its shared expert is
*ungated* (summed), matching ours; Qwen2-MoE gates the shared expert (sigmoid), which we'd have to
emulate.

## The crux: reconcile routing semantics (this is the real work, not the file writing)

A tensor rename alone produces wrong outputs unless the math matches. Known differences today:

| aspect | MoE Forge runtime | deepseek2 / qwen2moe | action |
|---|---|---|---|
| top-k weighting | **raw softmax prob** of selected experts (no renorm) | usually **renormalized** over the top-k (`norm_topk_prob`) | align one to the other |
| shared expert | **ungated**, weight 1 | deepseek2: ungated (+ optional scale); qwen2moe: sigmoid gate | use deepseek2 (ungated) |
| activation | SwiGLU (silu(gate)·up) | SwiGLU | matches ✓ |
| expert sizes | may be uneven | **stacked → must be uniform** | use `balanced_grouping` (have it) |

**Decision: align MoE Forge's MoE semantics to `deepseek2`** (set top-k normalization to match;
keep shared ungated), so *what we train is what llama.cpp runs*. Train-deploy parity beats
post-hoc fudging. This is a small change to `hf_runtime.py` routing + the recovery path.

## Tensor mapping (per layer; verify exact names against gguf-py constants)

- **Pass-through from the dense source** (unchanged): `token_embd`, `output_norm`, `output`,
  per-layer `attn_*` (q/k/v/o + norms), `ffn_norm`. Carve/upcycle only touch the FFN.
- **Router:** MoE Forge `token_router.weight/bias` → `blk.N.ffn_gate_inp`.
- **Routed experts (stacked, uniform):** carved `experts.{e}.{gate,up,down}` →
  `blk.N.ffn_{gate,up,down}_exps` (shape `[n_expert, ...]`). **Requires equal-size experts.**
- **Shared expert:** carved `shared.{gate,up,down}` → `blk.N.ffn_{gate,up,down}_shexp`.

## Metadata KVs (GGUF header)

`general.architecture = deepseek2`; from source: hidden size, n_layers, n_heads, n_kv_heads,
context length, rope params, vocab, norm eps. MoE-specific: `expert_count`,
`expert_used_count` (= top_k), `expert_shared_count` (= 1), `expert_feed_forward_length`
(= routed expert intermediate width — can be < the dense FFN, which is the carve case),
plus any required `expert_weights_scale` / norm flags to match the chosen weighting.

## Constraints / requirements

1. **Uniform routed expert sizes** — `balanced_grouping` already produces these (and our cheap
   tests showed balanced is the honest setting anyway). Uneven carves (old default) can't stack.
2. **Tokenizer** — emit the source tokenizer into the GGUF (gguf-py supports SPM/BPE vocab).
3. **Dependency** — add the `gguf` Python package (GGUFWriter). Today `gguf.py` only *reads*
   metadata; add a writer module.

## Correctness gate (non-negotiable)

After export, **verify numerical parity**: run the same prompt through (a) the MoE Forge HF
runtime and (b) llama.cpp on the GGUF, and confirm logits/next-token agree within tolerance.
Small semantic mismatches (router norm, shared scale, SwiGLU vs GeGLU) silently corrupt output —
this check is how we catch them. This is the make-or-break step.

## PMRA handoff

Once a valid GGUF MoE loads and passes parity, PMRA quantizes it directly (it already does
per-tensor mixed-rate allocation over GGUF). That realizes carve×PMRA / upcycle×PMRA and gives a
fast llama.cpp eval path — closing the loop the slow HF-wrapper benchmark has been blocking.

## Phased plan

0. **Pick + align arch (`deepseek2`).** Match MoE Forge routing semantics (top-k norm + ungated
   shared) to it; add a parity unit test (HF runtime == target semantics on synthetic weights).
1. **GGUF writer.** Map tensors + metadata for a balanced carved/upcycled wrapper → emit `.gguf`.
2. **llama.cpp parity check.** Load the GGUF, compare logits to the HF runtime (the gate above).
3. **PMRA.** Quantize the GGUF; confirm quality + measure the (now fast) eval.
4. **Tool wiring.** `convert --target gguf` emits the GGUF; document the carve×PMRA and
   upcycle×PMRA flows end-to-end.

## Risks

- **Semantic mismatch** with the chosen arch (the table above) — mitigated by aligning + the
  parity gate.
- **llama.cpp version drift** — pin a known-good build; arch metadata keys vary across versions.
- **Narrow experts** (`expert_feed_forward_length` < dense FFN) — supported by deepseek2 in
  principle; verify llama.cpp accepts our exact dims.
- **Upcycle path** maps more cleanly (full-width uniform experts) than carve (narrow experts), so
  if carve hits a GGUF snag, **upcycle→GGUF→PMRA is the lower-risk first target.**
