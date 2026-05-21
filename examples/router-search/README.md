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
# 1. Capture router-ready layers (adds hidden/gate/up to the grouping-search .npz):
python examples/grouping-search/capture_layer.py \
    --source-model outputs/smollm-moe-release-v5/wrapper/source-model \
    --tokenizer outputs/smollm-moe-release-v5/wrapper \
    --layer 3 --output examples/grouping-search/layer3.npz
#    (repeat for layers 9 = train, 6 = held-out validation)

# 2. Sanity-check the evaluator with no model, on a synthetic fixture:
PYTHONPATH=src python examples/router-search/make_fixture.py
PYTHONPATH=src python examples/router-search/eval_router.py \
    --candidate examples/router-search/candidates/seed_router.py \
    --layers examples/router-search/fixture.npz --experts 4 --top-k 2

# 3a. Evolve WITH an API key: alphaevolve.py calls the model to mutate candidates.
ANTHROPIC_API_KEY=... python tools/sonnet-evolve/alphaevolve.py --task tools/sonnet-evolve/tasks/router_evolve.json

# 3b. Evolve WITHOUT a key (the workflow used here): Claude Code spawns Sonnet subagents as
#     the generator -- each reads this prompt + the current best candidates and returns a new
#     router.py; you write them into candidates/ and score each with eval_router.py (pure
#     numpy, no key). Generation and evaluation stay separate: the model proposes, the
#     evaluator disposes. The candidates/ lineage (gen1_* hand-seeded, gen2_* sonnet-spawned)
#     is exactly such a run.

# 4. Validate the winner on the held-out layer:
PYTHONPATH=src python examples/router-search/eval_router.py \
    --candidate tools/sonnet-evolve/runs/router-evolve-*/best_candidate.py \
    --layers examples/grouping-search/layer6.npz --experts 8 --top-k 2
```

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
