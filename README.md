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
moe-forge plan <model-path> --goal balanced --output recipe.json
moe-forge profile <model-path> --text-file calibration.txt --output profile.json
moe-forge carve-manifest <model-path> --recipe recipe.json --profile profile.json --output carve-manifest.json
moe-forge carve-apply --manifest carve-manifest.json --output-dir carved-artifact
moe-forge carve-verify --manifest carve-manifest.json --artifact carved-artifact/carved-experts.safetensors
```

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
