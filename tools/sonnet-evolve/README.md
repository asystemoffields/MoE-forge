# sonnet-evolve

A small, reproducible, **task-agnostic** AlphaEvolve loop: claude-sonnet mutates the best code
candidates into new ones, a pluggable evaluator scores each, the best survive, repeat. Every
candidate, score, and prompt is logged to a run directory so a run is fully reproducible.

```
seed candidate ──► [ generate k mutations (sonnet) ] ──► [ score each (evaluator subprocess) ]
       ▲                                                              │
       └──────────────── keep the best as parents ◄──────────────────┘   × generations
```

The model proposes; the **evaluator disposes**. Generation is the only LLM step — selection is
a hard metric. That separation is the whole point: it grounds the search in reality instead of
in agents judging agents.

## Run it

```bash
export ANTHROPIC_API_KEY=...        # PowerShell: $env:ANTHROPIC_API_KEY = "..."
python tools/sonnet-evolve/alphaevolve.py --task tools/sonnet-evolve/tasks/carve_grouping.json
# overrides: --generations 5 --candidates-per-gen 4 --run-dir <dir>
```

It prints each candidate's score and writes `tools/sonnet-evolve/runs/<task>-<ts>/` with
per-generation candidate files, `manifest.json` (full history + best), and `best_candidate.py`.

Cost note: each generation makes `candidates_per_gen` sonnet calls; the stable task prompt is
sent as a cached system block, so repeated generations reuse it.

## Adopt it to a new task — three files, no framework changes

1. **An evaluator** — any command `... --candidate <file>` that prints JSON containing your
   score (e.g. `{"mean_error": 0.41}`). It defines what "good" means; design it carefully — the
   loop will exploit any weakness in the metric (validate winners on held-out data the loop
   never saw).
2. **A seed candidate** — a working code file the loop starts from and tries to beat.
3. **A task JSON config** — paths + score key + direction + the generation prompt:

```json
{
  "name": "my-task",
  "evaluator_cmd": ["python", "score.py", "--candidate", "{candidate}"],
  "evaluator_env": {"PYTHONPATH": "src"},
  "score_key": "loss",
  "direction": "min",
  "seed_candidate": "seed.py",
  "prompt_file": "prompt.md",
  "candidate_filename": "candidate.py",
  "generations": 3,
  "candidates_per_gen": 3,
  "parents_kept": 2
}
```

`{candidate}` in `evaluator_cmd` is replaced with each candidate file's path. `direction` is
`min` or `max`. Bad candidates (evaluator prints `{"error": ...}` or crashes) score worst and
are never selected, so the loop is robust to malformed generations.

## Included task: carve-grouping

`tasks/carve_grouping.json` evolves a MoE Forge carve channel-grouping function, scored by
`examples/grouping-search/eval_candidate.py` (oracle-top-k reconstruction error on held-back
SmolLM layers). It seeds from the co-activation-clustering baseline and tries to beat it. See
`examples/grouping-search/` for capturing the layer `.npz` files the evaluator needs.

## Reproducibility & honesty

- `manifest.json` records the model, direction, per-candidate scores, and the winning file —
  re-running the evaluator on any candidate reproduces its score.
- The generation step is LLM sampling, so candidates vary run-to-run; the *evaluation* is
  deterministic given the evaluator and its inputs.
- **Goodhart guard:** hold out validation data the evaluator does not see during the loop, and
  re-score the winner on it. A win that doesn't generalize is the loop exploiting the metric.
