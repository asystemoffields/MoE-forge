# Real HF Smoke Recipe

This recipe runs the carved-MoE path on a small local Hugging Face checkpoint with tokenizer-backed text samples. It is intended for checkpoints such as `HuggingFaceTB/SmolLM-135M` or another local Llama/Qwen/Gemma-style causal LM folder.

The default `recipe.json` uses `moe_layers: "all"` so every mapped FFN layer is converted into carved shared/expert tensors.

Run these commands from this directory after installing the HF extras:

```powershell
pip install -e "..\..[hf]"
$model = "C:\models\SmolLM-135M"
moe-forge inspect $model --json
moe-forge carve-manifest $model --recipe recipe.json --output carve-manifest.json
moe-forge carve-apply --manifest carve-manifest.json --output-dir carved-artifact
moe-forge carve-verify --manifest carve-manifest.json --artifact carved-artifact/carved-experts.safetensors --output carve-verify-report.json
moe-forge wrapper-export --manifest carve-manifest.json --artifact carved-artifact/carved-experts.safetensors --router-plan router-plan.json --token-router-top-k 2 --copy-artifact --copy-source-model --output-dir wrapper
```

Copy the template configs and replace `<LOCAL_HF_CHECKPOINT>` with the same checkpoint path:

```powershell
Copy-Item eval-batch-text.template.json eval-batch-text.json
Copy-Item recovery-experiment-text-3step.template.json recovery-experiment-text-3step.json
moe-forge eval-batch --config eval-batch-text.json
moe-forge recovery-experiment --config recovery-experiment-text-3step.json
moe-forge model-card --wrapper wrapper --eval-report eval-runs-text/eval-learned_router.json --recovery-report recovery-experiment-text-3step/recovery-experiment-report.json --validation-report recovery-experiment-text-3step/recovered-wrapper-validation.json --output MODEL_CARD.md
```

Check the wrapper through native Transformers loading:

```powershell
python -c "import moeforge; from transformers import AutoModelForCausalLM; m=AutoModelForCausalLM.from_pretrained('wrapper'); print(type(m).__name__, [r.layer for r in m.replacement_report.replaced])"
```

Expected lab-notebook artifacts:

- `carve-verify-report.json`
- `eval-runs-text/eval-batch-manifest.json`
- `eval-runs-text/eval-compare.html`
- `recovery-experiment-text-3step/recovery-experiment-report.json`
- `recovery-experiment-text-3step/recovery-experiment.html`
- `recovery-experiment-text-3step/recovered-wrapper-validation.json`
- `recovery-experiment-text-3step/recovered-wrapper/recovery-export-report.json`
- `recovery-experiment-text-3step/recovered-wrapper/learned-router.safetensors` when router recovery is enabled
- `MODEL_CARD.md`

The reports and model card record text-file SHA-256 provenance, active expert selections, learned-router token counts, latency ratios, teacher-KL and next-token NLL deltas, recovered tensor metadata, checkpoint identity, reproduction commands, and recovered-wrapper validation evidence.
