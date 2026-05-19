# Tiny HF Smoke Recipe

This recipe creates a tiny local Llama-style checkpoint with a toy tokenizer and runs the carved-MoE path end to end. The default eval and recovery samples come from `token-ids.json`, so the generated reports record dataset file identity instead of only inline samples.

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

The eval and recovery reports include logits parity, teacher-KL, next-token NLL, active experts, latency ratios, recovered tensor metadata, and SHA-256 provenance for `token-ids.json`. `commands.json` captures the runnable command recipe and config list for lab notebook review.

For a longer local run, use `recovery-experiment-longer.json`. It keeps the same tiny model and sample style, but runs 20 recovery steps and records extra checkpoints:

```powershell
moe-forge recovery-experiment --config recovery-experiment-longer.json
```

The generated checkpoint can also run tokenizer-backed text-file eval and recovery from `sample-prompts.txt`:

```powershell
moe-forge eval-batch --config eval-batch-text.json
moe-forge recovery-experiment --config recovery-experiment-text.json --max-steps 2
moe-forge recovery-compare recovery-experiment/recovery-experiment-report.json recovery-experiment-text/recovery-experiment-report.json --output recovery-compare.json --html-output recovery-compare.html
```

For a longer text-file recovery run:

```powershell
moe-forge recovery-experiment --config recovery-experiment-text-longer.json
```
