# Task: invent a better way to compress an LLM — you may RECONCEIVE what compression means

You write a complete compression *method* as code; a separate hermetic verifier scores it. You
are explicitly free to reconceive the problem: **the weights are not sacred — you do not have to
preserve them.** Preserve whatever makes held-out quality-per-byte best.

## Contract
```python
def compress(model, calib_tokens, budget_bytes) -> bytes      # the ONLY per-model payload
def decompress(artifact: bytes) -> model                      # returns a runnable causal-LM nn.Module
```
- `model`: a loaded HF causal LM. Proving model = SmolLM-135M (Llama arch, 30 layers, hidden 576,
  intermediate 1536, ships bf16). You may read its weights/config and RUN it on calib.
- `calib_tokens`: int tensor `[n, seq]` you may study (collect activations, gradients, anything).
- `decompress` receives ONLY the artifact — no model, no calib, no globals. Reconstruct the
  architecture from data you stored (the config is tiny; store it — see the seeds).

## How you are judged (the only things that count)
- `shipped_bytes = len(artifact)`
- `quality = held-out NLL of decompress(compress(model))`  (held-out text is hidden, disjoint from calib)
- You win by getting **lower NLL at fewer bytes**. Baselines on SmolLM-135M (ratio vs bf16):
  lossless no-op = **1.00× / +0.000 NLL**; int8 round-to-nearest = **1.65× / +0.096 NLL**.
  Beat the int8 point, or stake out a new point on the frontier (e.g. far smaller with graceful loss).

## You may reconceive what to preserve (encouraged)
- preserve the model's **behavior/outputs** on calib, not its weights (calibration-fit, in-place
  distillation, activation matching) — then the weights can change however you like;
- preserve what matters for the **output** (sensitivity / Hessian / Jacobian weighting), not raw MSE;
- change the **representation** (rotate / PCA / SVD weights or activations into a cheaper basis), store
  few coefficients, reconstruct;
- exploit **cross-layer / cross-tensor redundancy** (shared codebooks, joint factorization, delta-coding
  similar layers, tying you discover from calib);
- **entropy-code** the result — the artifact's real bytes are what count, so coding the symbols is free quality;
- **non-uniform** schemes — spend bytes where the model is sensitive, starve where it isn't;
- something with no name yet.

## Rules (verifier-enforced; violations score worst)
- numpy + torch + transformers only; NO network, NO disk, NO file reads.
- `compress` must finish in ~2 minutes on CPU for a 135M model (calibration optimization is fine;
  full retraining is too slow).
- Everything needed at decode time must live in the artifact (it costs bytes). `decompress` gets nothing else.
- `decompress` must return a runnable `nn.Module` (rebuild via `AutoConfig.for_model` + `from_config`,
  exactly like the seeds). Be deterministic; return finite weights.

## Output
Return ONLY a single ```python code block defining `compress` and `decompress`. No prose.
