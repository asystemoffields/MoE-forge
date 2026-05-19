# Tiny HF Smoke Recipe

This recipe creates a tiny local Llama-style checkpoint and runs the carved-MoE path end to end.

Run these commands from this directory after installing the HF extras:

```powershell
pip install -e "..\..[hf]"
python make_tiny_llama.py --output tiny-llama
moe-forge carve-manifest tiny-llama --recipe recipe.json --output carve-manifest.json
moe-forge carve-apply --manifest carve-manifest.json --output-dir carved-artifact
moe-forge carve-verify --manifest carve-manifest.json --artifact carved-artifact/carved-experts.safetensors --output carve-verify-report.json
moe-forge wrapper-export --manifest carve-manifest.json --artifact carved-artifact/carved-experts.safetensors --router-plan router-plan.json --copy-artifact --output-dir wrapper
moe-forge eval-batch --config eval-batch.json
moe-forge recovery-experiment --config recovery-experiment.json --max-steps 2
moe-forge recovery-validate --source-wrapper wrapper --recovered-wrapper recovery-experiment/recovered-wrapper --checkpoint recovery-experiment/recovery/checkpoints/checkpoint-step-2.json --output recovery-experiment/recovered-wrapper-validation-cli.json
moe-forge smoke-assert --run-dir . --output smoke-assertions.json
```

Expected lab-notebook artifacts:

- `eval-runs/eval-batch-manifest.json`
- `eval-runs/eval-compare.html`
- `recovery-experiment/recovery-experiment-report.json`
- `recovery-experiment/recovery-experiment.html`
- `recovery-experiment/recovered-wrapper-validation.json`
- `smoke-assertions.json`

The eval and recovery reports include logits parity, teacher-KL, next-token NLL, active experts, latency ratios, and recovered tensor metadata.

For a longer local run, use `recovery-experiment-longer.json`. It keeps the same tiny model and sample style, but runs 20 recovery steps and records extra checkpoints:

```powershell
moe-forge recovery-experiment --config recovery-experiment-longer.json
```
