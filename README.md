# MoE Forge

MoE Forge is a Python tool for turning dense language models into custom Mixture-of-Experts candidates.

It is designed as a research-lab quality tool with a low-friction path in:

- a student experimenting on a laptop
- an engineer trying a model surgery recipe on one GPU
- a lab running controlled ablations across architectures

The project goal is a practical workflow:

```powershell
moe-forge inspect C:\models\gemma
moe-forge plan C:\models\gemma --goal balanced --experts 8 --top-k 2
moe-forge run recipe.json
```

The first implementation slice focuses on reliable inspection and recipe planning. Conversion backends can build on the same recipe format.

## Current Commands

```powershell
moe-forge adapters
moe-forge inspect <model-path> --json
moe-forge preflight --model <model-path> --recipe recipe.json --wrapper wrapper --output preflight-report.json
moe-forge plan <model-path> --goal balanced --output recipe.json
moe-forge plan <model-path> --moe-layers all --output whole-model-recipe.json
moe-forge profile <model-path> --text-file calibration.txt --output profile.json
moe-forge router-plan --profile profile.json --pool-size 2 --output router-plan.json
moe-forge carve-manifest <model-path> --recipe recipe.json --profile profile.json --output carve-manifest.json
moe-forge carve-apply --manifest carve-manifest.json --output-dir carved-artifact
moe-forge carve-verify --manifest carve-manifest.json --artifact carved-artifact/carved-experts.safetensors
moe-forge wrapper-export --manifest carve-manifest.json --artifact carved-artifact/carved-experts.safetensors --router-plan router-plan.json --token-router-top-k 2 --copy-source-model --output-dir wrapper
moe-forge eval-hf <model-path> --wrapper wrapper --output eval-report.json
moe-forge eval-report-html --input eval-report.json --output eval-report.html
moe-forge eval-compare eval-all.json eval-router.json --output eval-compare.json --html-output eval-compare.html
moe-forge eval-batch --config eval-batch.json --output-dir eval-runs
moe-forge recovery-plan --config recovery.json --output recovery-plan.json
moe-forge recovery-run --plan recovery-plan.json --output recovery-run-report.json
moe-forge recovery-export --checkpoint checkpoints/checkpoint-step-100.json --wrapper wrapper --output-dir recovered-wrapper
moe-forge recovery-validate --source-wrapper wrapper --recovered-wrapper recovered-wrapper --checkpoint checkpoints/checkpoint-step-100.json
moe-forge recovery-experiment --config recovery-experiment.json --output-dir recovery-experiment
moe-forge recovery-compare recovery-a.json recovery-b.json --output recovery-compare.json --html-output recovery-compare.html
moe-forge model-card --wrapper wrapper --eval-report eval-report.json --recovery-report recovery-experiment/recovery-experiment-report.json --validation-report recovery-experiment/recovered-wrapper-validation.json --output MODEL_CARD.md
moe-forge smoke-assert --run-dir . --output smoke-assertions.json
```

For an end-to-end local smoke run, see [examples/tiny-hf-smoke](examples/tiny-hf-smoke). For a tokenizer-backed small real checkpoint recipe, see [examples/real-hf-smoke](examples/real-hf-smoke).

Supported inputs:

- Hugging Face model folders with `config.json`
- GGUF files with readable metadata
- Hugging Face model ids such as `google/gemma-4-E2B-it` or `hf:org/model@revision`

Current inspection includes:

- architecture adapter detection
- dense/MoE status
- layer, hidden, FFN, vocabulary, attention, and context metadata
- local checkpoint file detection
- safetensors header indexing without loading full tensors
- expected FFN gate/up/down tensor-map validation where an adapter is known

Current profiling includes:

- optional Torch/Transformers HF backend
- hook-based FFN `gate`, `up`, and `down` activation capture
- per-channel mean absolute activation, RMS, active rate, and positive rate
- top-channel summaries with optional full vectors
- per-document activation summaries with stable text hashes for EMO-style expert-pool analysis
- first-pass document expert-pool recommendations for selected-subset routing experiments
- first-pass shared/routed expert channel assignments from activation importance

Current routing support includes:

- EMO-inspired `document_pool_then_token_router` metadata
- default expert-pool fallback from aggregate document scores
- expert-pool selection by document index, text hash, or raw text hash
- runtime integration for selected-subset carved MLP execution

Current wrapper support includes:

- `config.json` export for `MoEForgeConfig.from_package(...)`
- `moeforge_config.json` package export for carved FFN artifacts
- native `AutoModelForCausalLM.from_pretrained(...)` loading for installed MoE Forge packages
- optional router-plan packaging
- optional learned per-token top-k router modules for native HF wrapper runs
- reloadable carved layer runtime from wrapper config
- PyTorch module loading for carved FFN layer parity and selected expert subsets
- in-place FFN replacement for tiny HF causal-LM parity checks

Current evaluation support includes:

- dense-vs-carved HF logits parity reports
- per-sample max/mean absolute error and latency
- per-layer dense/all-expert/selected-expert attribution
- replacement metadata, active expert records, memory notes, warnings, and package metadata
- teacher-KL, dense next-token NLL, carved next-token NLL, NLL deltas, and loss-token counts
- all-expert, default-pool, and document-router expert modes for routed subset tradeoff runs
- learned-router eval mode with per-layer token counts, top-k, expert token counts, and selected-weight summaries
- learned-router route-pattern introspection with unique route counts, entropy, probability mass, and compact route-token summaries
- self-contained HTML reports from eval JSON artifacts
- multi-report comparison JSON/HTML for quality-first ranking, speed ratios, and active expert summaries
- config-driven eval batches that run multiple expert modes, emit per-mode reports, compare completed runs, and preserve recovery-eval settings
- teacher-KL recovery plan artifacts with loss, optimizer, sample, checkpoint, and before/after eval-batch comparison records
- a tiny recovery runner that consumes recovery plans, computes teacher-KL/logits losses on input-id batches, promotes carved tensors for training, and writes checkpoint metadata
- recovery checkpoint export that applies trainable tensor state back into a recovered wrapper package
- router-only recovery export that writes learned router state to `learned-router.safetensors`
- recovered-wrapper validation that reloads package metadata, checks checkpoint/export compatibility, validates router safetensors, proves native AutoModel loading, and compares original vs recovered safetensors metadata
- recovery experiment orchestration that runs before/after eval batches around recovery and writes JSON/HTML comparison reports with validation and router-export evidence
- Markdown model-card generation from wrapper metadata plus eval, router-activity, recovery, router-export, validation, and reproduction-command artifacts
- agent-friendly preflight JSON reports with readiness checks, blockers, warnings, and suggested next commands
- smoke assertions that verify tiny HF recipe artifacts, quality metrics, recovered-wrapper validation, and report links

Example eval batch config:

```json
{
  "model": "C:/models/tiny-llama",
  "wrapper": "wrapper",
  "output_dir": "eval-runs",
  "expert_modes": ["all", "default-pool", "router"],
  "input_ids": [[1, 2, 3, 4]],
  "write_html": true,
  "recovery_eval": {
    "enabled": true,
    "metrics": ["logits_parity", "teacher_kl"]
  }
}
```

Example native HF load:

```python
import moeforge
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("wrapper")
tokenizer = AutoTokenizer.from_pretrained(model.config.source_model)
outputs = model(**tokenizer("Dense to MoE", return_tensors="pt"))
```

Example recovery plan config:

```json
{
  "teacher_model": "C:/models/tiny-llama",
  "student_model": "C:/models/tiny-llama",
  "wrapper": "wrapper",
  "output_dir": "recovery-run",
  "train": { "text_file": "train.txt", "sequence_length": 128 },
  "eval": { "input_ids": [[1, 2, 3, 4]] },
  "loss": { "teacher_kl_weight": 1.0, "temperature": 2.0 },
  "schedule": { "steps": 100, "eval_every_steps": 25 },
  "before_eval_batch": "eval-before/eval-batch-manifest.json",
  "after_eval_batch": "eval-after/eval-batch-manifest.json"
}
```

Example recovery experiment config:

```json
{
  "model": "C:/models/tiny-llama",
  "wrapper": "wrapper",
  "output_dir": "recovery-experiment",
  "strict_validation": true,
  "eval": {
    "expert_modes": ["all", "default-pool", "router"],
    "input_ids": [[1, 2, 3, 4]],
    "write_html": true
  },
  "train": { "input_ids": [[1, 2, 3, 4]] },
  "recovery": {
    "loss": { "teacher_kl_weight": 1.0, "temperature": 2.0 },
    "schedule": { "steps": 100, "save_every_steps": 25 }
  }
}
```

Current carving support includes:

- validated carve manifests for local HF checkpoints
- source tensor, source file, tensor shape, and channel-axis records
- shared/routed expert channel assignments per layer
- warnings for incomplete tensor maps, approximate widths, duplicates, missing channels, and axis ambiguity
- safetensors materialization for carved shared/expert tensor artifacts
- reconstruction verification that carved tensors exactly rebuild the source FFN weights
- all-experts PyTorch runtime for carved gated MLP parity checks

Current architecture adapters:

- Llama/Mistral-style gated MLPs
- Qwen2/Qwen3-style gated MLPs
- Gemma-family gated MLPs
- Phi-family fused MLPs for adapter-MoE planning

## Design

MoE Forge separates user intent from conversion details.

User intent includes:

- quality, speed, tiny, or balanced goal
- target export format
- hardware budget
- expert count and top-k routing preference
- shared expert ratio
- layer range

The planner converts that intent into a recipe:

- strategy: carved MLP, sparse upcycle, adapter-MoE, or inspection-only
- MoE layer selection
- router initialization method
- calibration and recovery settings
- validation checks
- export target

## Quality Bar

Every run should be reproducible, inspectable, and measurable.

- Recipes are saved as explicit JSON artifacts.
- Model inspection records source format, architecture, layer shape, dense/MoE status, and warnings.
- Conversion steps should emit manifests for tensor mappings, router initialization, calibration data, and export decisions.
- Evaluation should compare the dense source and MoE candidate on perplexity, teacher KL, smoke prompts, speed, memory, active compute, and router balance.
- Defaults should work on modest hardware, while advanced users can override layout, routing, training, calibration, and export choices.
- Experimental methods should be implemented as plugins or small modules so new papers can be reproduced without reshaping the whole tool.

## Roadmap

1. Architecture inspection and recipe planning
2. Activation profiling for dense FFN layers
3. Carved-MLP expert construction for HF safetensors
4. Analytical router initialization
5. Recovery training against the dense teacher
6. Evaluation reports for perplexity, KL, active compute, and speed
7. GGUF export once runtime layout support is available

See [RESEARCH_NOTES.md](RESEARCH_NOTES.md) for research threads guiding router and modularity design.
