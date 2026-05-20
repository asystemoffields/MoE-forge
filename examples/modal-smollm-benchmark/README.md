# Modal SmolLM Benchmarks

This example runs the benchmark plan emitted by `moe-forge benchmark-plan` on Modal.

Use it when local CPU benchmarking is too slow and you want the dense source checkpoint and
the MoE wrapper tested with the same LightEval checkout, task list, sample cap, and batch size.
The runner pins LightEval to `v0.12.2` because the SmolLM/Cosmopedia custom task file was written
against that task API.

## Base SmolLM

```powershell
$env:PYTHONPATH="src"
python -m moeforge benchmark-plan `
  --source-model HuggingFaceTB/SmolLM-135M `
  --moe-model /vol/smollm-moe-v5 `
  --suite smollm-base `
  --output outputs/smollm-moe-release-v5/benchmark-plan.json `
  --max-samples 1000 `
  --batch-size 16

modal volume create moeforge-benchmarks
modal volume put moeforge-benchmarks outputs/smollm-moe-release-v5/recovery-experiment/recovered-wrapper smollm-moe-v5

modal run examples/modal-smollm-benchmark/modal_lighteval.py `
  --plan outputs/smollm-moe-release-v5/benchmark-plan.json `
  --run-name smollm-base-v5 `
  --which both

modal volume get moeforge-benchmarks /runs/smollm-base-v5 outputs/modal-smollm-base-v5
```

After the run, pass the dense and MoE result JSONs to:

```powershell
$env:PYTHONPATH="src"
python -m moeforge benchmark-compare `
  --dense-report outputs/modal-smollm-base-v5/dense/results.json `
  --moe-report outputs/modal-smollm-base-v5/moe/results.json `
  --suite smollm-base `
  --output outputs/modal-smollm-base-v5/benchmark-compare.json
```

## SmolLM Instruct

If the base MoE meets the benchmark retention gate, forge the instruct checkpoint and run the
instruct suite head-to-head:

```powershell
$env:PYTHONPATH="src"
python -m moeforge benchmark-plan `
  --source-model HuggingFaceTB/SmolLM-135M-Instruct `
  --moe-model /vol/smollm-instruct-moe `
  --suite smollm-instruct `
  --output outputs/smollm-instruct-benchmark-plan.json `
  --max-samples 1000 `
  --batch-size 16
```

The instruct plan enables chat-template evaluation and includes IFEval plus MT-Bench metadata.
MT-Bench needs judge-model configuration and raw judgment artifacts before it should be used as
release evidence.
