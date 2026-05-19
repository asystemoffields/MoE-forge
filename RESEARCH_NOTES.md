# Research Notes

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
