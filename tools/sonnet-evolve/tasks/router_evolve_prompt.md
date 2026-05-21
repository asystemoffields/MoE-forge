# Task: evolve a training-free top-k router for a carved MoE

A dense FFN layer has been carved into a **shared** channel group (always active) plus **E
disjoint expert groups** over the intermediate channels. The grouping is FIXED. At inference,
only **top-k of the E experts** fire per token. Your job is the **router**: the rule that, per
token, decides *which* k experts fire — seeing only the FFN input hidden state, not the
experts' actual outputs (computing those is the dense work carving avoids).

Because the experts are disjoint channel sets, the dense output is exactly
`shared + sum_e expert_e`. Selecting k experts reconstructs `shared + sum_{e in top-k} expert_e`;
the dropped experts are the error. We score the **relative reconstruction error** vs the dense
FFN (mean over tokens of `||dense - reconstruction|| / ||dense||`) — the SAME metric the carve
grouping search uses. An **oracle** that selects by each expert's true output norm is the floor;
**random** selection is the ceiling. Drive `mean_error` down toward the oracle. Minimize it.

## Contract — define exactly these two functions (numpy only)

```python
def build_router(ctx, n_experts, top_k, rng):
    """OFFLINE. Build cheap routing parameters from FIXED weights + a calibration token split.
    ctx["calib_hidden"]:      [Tc, H]  calibration FFN input hidden states
    ctx["calib_activations"]: [Tc, I]  calibration gated activations  a = silu(h@gate^T) * (h@up^T)
    ctx["gate"]: [I, H]   ctx["up"]: [I, H]   ctx["down"]: [H, I]   (the FFN weight matrices)
    ctx["assignment"]: [I] ints  (-1 = shared/always-active, else expert id 0..n_experts-1)
    ctx["importance"]: [I]  per-channel mean |activation|
    Return: dict[str, np.ndarray]. Total element count must be <= 8 * n_experts * H.
    """

def route(hidden, state, n_experts, top_k):
    """PER TOKEN. Return scores [hidden.shape[0], n_experts]; higher = more likely selected.
    Sees ONLY eval hidden states [T, H] and the state you built. Must be finite and cheap
    (O(n_experts * H) per token, e.g. a few dot products) — NOT a full re-projection."""
```

`build_router` may do heavy offline analysis (SVD, clustering, a closed-form least-squares fit
of hidden -> per-expert energy on the calibration split, etc.) but must COMPRESS what it learns
into a small `state` (the budget = `8 * n_experts * H` floats forbids storing the full gate
matrix and recomputing activations — that would be the dense path, not a router). `route` then
uses only `hidden` and `state`.

## Signal to exploit (why the seed leaves error on the table)
The seed scores each expert by `hidden . (mean gate row of its channels)`. That ignores:
- the **silu** nonlinearity (only positive gate pre-activations pass through) and the
  multiplicative **up**-projection gate — both shape which channels actually fire;
- **down**-row magnitudes — channels with large `||down[:, i]||` dominate the OUTPUT, so an
  expert's routing importance is about output contribution, not raw activation;
- **multimodality** — one key per expert is weak if an expert's channels co-fire in several
  distinct hidden-state regimes (consider a few sub-keys per expert, scored by max or sum);
- **magnitude vs direction** — the oracle ranks by contribution *norm*; predicting that
  magnitude (e.g. fitting hidden -> expert energy on the calibration split) can beat a pure
  direction match;
- **centering / whitening** the hidden states, and per-expert score calibration (bias/scale).

## Rules
- Return finite scores of the exact shape `[T, n_experts]`. Be deterministic.
- Stay within the state budget. Do not import torch or read files; numpy only.
- A held-out layer (never seen during the loop) re-scores the winner — solutions that only
  work in-sample are the loop exploiting the metric, so prefer robust, general routing rules.
