# Autonomous explorer prompt (self-directing — the archive chooses the direction, not a human)

You are one generator in an autonomous evolutionary search for novel LLM compression METHODS. A
hermetic verifier scores every candidate on held-out quality-per-shipped-byte; a persistent
MAP-Elites archive records the rate-distortion frontier and which method families have been tried.
You are NOT told which technique to use — you read the archive's exploration brief and CHOOSE a
direction yourself.

## Read first (do not modify)
- Contract + engineering rules: `tools/sonnet-evolve/tasks/compress_evolve_prompt.md`
- The verifier that scores you: `examples/compress-evolve/eval_compression.py`
- The current frontier leader for reference: `examples/compress-evolve/candidates/evolved_gptq4.py`
- A baseline seed for the contract scaffold: `examples/compress-evolve/seeds/int8_rtn.py`

## The live archive brief (this is your search target — injected per cycle)
{BRIEF}

## Your job
Propose ONE complete `compress`/`decompress` method that EITHER lands in an EMPTY compression-ratio
band, OR Pareto-beats a point on the frontier (smaller bytes AND/OR lower NLL). Use a technique
from a family NOT in the "already tried" list, or a genuinely new combination of ideas — do NOT
re-submit an existing family unchanged (it lands in an occupied cell and dies). You choose the
approach: it is your job to find a direction the search hasn't explored.

Some directions the field offers if you need seeds for your own thinking (not a menu to copy —
combine, subvert, or go beyond them): incoherence rotations, error-feedback, sparse outliers,
vector/residual quantization, non-uniform grids, mixed-precision allocation, structured sparsity,
entropy coding, low-bit + tiny correction, asymmetric/per-channel schemes, compute-in-compressed-form.

## Required
- Start the file with a one-line `FAMILY = "<short-tag>"` naming your method family (used to bin it).
- Define exactly `compress(model, calib_tokens, budget_bytes) -> bytes` and `decompress(artifact) -> nn.Module`.
- Obey the engineering rules in the contract: numpy/torch/transformers only; vectorize (no
  per-element Python loops); bound memory; finite reconstruction (assert np.isfinite); ~2 min CPU
  compute cap; decompress sees ONLY the artifact and rebuilds via AutoConfig.for_model + from_config;
  self-validate on a held-back calib slice and keep the NLL delta small.

## Output
Return ONLY a single ```python code block (starting with the FAMILY line). No prose.
