# router-search

Evolve the **top-k router** of a carved MoE, the way `grouping-search/` evolved the channel
grouping. The carve's central finding is that ~100% of the quality gap is the sparse top-k
router; gradient recovery of the router plateaus. This asks the complementary question with a
cheap, training-free evaluator: **is there a better expert-selection rule than the learned
linear-softmax top-k?**

## What it measures
The grouping is held FIXED (`balanced_grouping`). A candidate router scores the experts per
token from the hidden state alone; the evaluator selects top-k by that score and reports the
relative reconstruction error vs the dense FFN — the *same* metric as
`moeforge.grouping.oracle_topk_error`. Two reference points come for free:
- **oracle** = selection by each expert's true output norm (the floor a perfect router hits);
- **random** = random selection (the ceiling).

A candidate is two functions — `build_router` (offline, may fit on a calibration token split,
returns a small `state`) and `route` (per token, sees only hidden + state). A **state-size
budget** (`8 * n_experts * H` floats) keeps the router cheap and blocks the Goodhart exploit of
recomputing the full activations to reproduce the oracle. See the contract in
`../../tools/sonnet-evolve/tasks/router_evolve_prompt.md`.

## Run it
```bash
# 1. Capture router-ready layers (hidden/gate/up/down) from the dense base model. A hub id or
#    a local path both work; layers 3 + 9 = train, 6 = held-out validation. (~6MB each, gitignored.)
for L in 3 6 9; do
  PYTHONPATH=src python examples/grouping-search/capture_layer.py \
      --source-model HuggingFaceTB/SmolLM-135M --tokenizer HuggingFaceTB/SmolLM-135M \
      --layer $L --output examples/router-search/layer$L.npz
done

# 2. Sanity-check the evaluator with no model, on a synthetic fixture:
PYTHONPATH=src python examples/router-search/make_fixture.py
PYTHONPATH=src python examples/router-search/eval_router.py \
    --candidate examples/router-search/candidates/seed_router.py \
    --layers examples/router-search/fixture.npz --experts 4 --top-k 2

# 3a. Evolve WITH an API key: alphaevolve.py calls the model to mutate candidates.
ANTHROPIC_API_KEY=... python tools/sonnet-evolve/alphaevolve.py --task tools/sonnet-evolve/tasks/router_evolve.json

# 3b. Evolve WITHOUT a key (the workflow used here): Claude Code spawns Sonnet subagents as
#     the generator -- each reads the prompt + current best candidates and returns a new
#     router.py; you write them into candidates/ and score each with eval_router.py (pure
#     numpy, no key). Generation and evaluation stay separate: the model proposes, the
#     evaluator disposes. The kept winner evolved_energy.py came from exactly such a run
#     (gen* scratch is gitignored); see Result below.

# 4. Validate the winner on the held-out layer:
PYTHONPATH=src python examples/router-search/eval_router.py \
    --candidate examples/router-search/candidates/evolved_energy.py \
    --layers examples/router-search/layer6.npz --experts 8 --top-k 2
```

## Result (real captured SmolLM-135M layers)
A key-free run -- four Sonnet subagents proposed, `eval_router.py` disposed -- produced
`candidates/evolved_energy.py`, a second-order energy estimator that wins at every operating
point and generalizes to the held-out layer:

| setting              | oracle | random | evolved_energy | seed   |
|----------------------|--------|--------|----------------|--------|
| train(L3+L9) 8/top2  | 0.5545 | 0.6194 | **0.5942**     | 0.5958 |
| held-out L6  8/top2  | 0.6006 | 0.6791 | **0.6426**     | 0.6482 |
| train(L3+L9) 4/top2  | 0.4473 | 0.5041 | **0.4825**     | 0.4837 |

Two honest takeaways: (1) the search works — `evolved_energy` beats the gate-direction seed and
every other candidate everywhere, and the win holds out-of-sample; the *same* rule ranked LAST
on the synthetic fixture, so only real layers can rank routers. (2) Routing headroom is modest:
the best rule closes only ~40% of the random->oracle gap (~0.035-0.042 relative error left to
the oracle). The dominant layer-reconstruction loss is the sparsity/grouping, not the selection
rule — consistent with the sparsity-frontier finding. So a better router is best understood as a
better recovery **warm-start**, not a way around carve sparsity.

## Honesty / scope
This proxy scores **selection** on disjoint experts, so per-expert mixing *weights* do not enter
the reconstruction (a selected expert contributes its full output). Softmax-weight effects and
train==deploy interactions are validated downstream via teacher-KL on the real wrapper. A win
here means a better *selection rule*; promote it by deriving the router init/parameterization
from the evolved rule, then confirm with a recovery run.

The synthetic `fixture.npz` (random Gaussian weights, ~40 tokens) only validates the *mechanism*
and that the evaluator discriminates (random ~0.52 vs oracle ~0.26); it cannot rank routers,
because the calibration-fit and structure-aware rules need real captured layers to show their
edge. Run the search on captured SmolLM layers for any real conclusion.

## Tests
`PYTHONPATH=src python -m pytest examples/router-search/test_eval_router.py` — metric parity
with the library oracle, seed lands between oracle and random, budget guard fires.
