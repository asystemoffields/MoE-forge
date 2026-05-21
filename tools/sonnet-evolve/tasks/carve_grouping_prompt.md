You are evolving a Python function that partitions a dense FFN layer's intermediate channels into a shared group plus expert groups for Mixture-of-Experts compression. Your function is scored by the **oracle-top-k reconstruction error** (lower is better): the grouping is given a perfect router (per token, the shared channels plus the top-k expert groups by contribution are active), and the error is the relative reconstruction vs the dense FFN, averaged over tokens.

Write a function with EXACTLY this signature:

```python
import numpy as np
SHARED = -1

def group(ctx, n_experts, shared_ratio, rng):
    # ctx["importance"]: float64 ndarray [I] — per-channel mean |gated activation|
    # ctx["activations"]: float64 ndarray [T, I] — per-token gated activations (co-activation signal)
    # returns: int ndarray [I]; SHARED (-1) = always-active channel, else expert id 0..n_experts-1
    ...
```

Constraints:
- numpy only; deterministic given `rng`; each call must run in well under a minute (I ≈ 1536 channels, T ≈ 200 tokens).
- The returned assignment must have shape [I] and every value must be SHARED or in 0..n_experts-1.
- Scoring regime: n_experts=8, shared_ratio=0.125, top_k=2.

Why groupings score well: the metric rewards placing channels that fire together (for the same tokens) into the same expert, so few experts reconstruct most of each token's output. High-importance channels are good candidates for the always-active SHARED group. Co-activation structure lives in `ctx["activations"]` (correlations / clustering of channel activation vectors). Balanced expert sizes can help because whole experts are dropped per token.

Ideas worth exploring: correlation/cosine clustering of channel activation vectors, importance-weighted distances, balanced (equal-size) clustering, spectral/graph partitioning of a co-activation similarity matrix, NMF/SVD latent-component grouping, or local refinement that directly lowers the oracle-top-k error. Beat the parent.
