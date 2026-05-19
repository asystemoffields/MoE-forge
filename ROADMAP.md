# MoE Forge Roadmap

MoE Forge should feel approachable on a laptop and credible in a research lab.

## Product Shape

The tool has three layers:

1. Friendly CLI for common workflows.
2. Explicit recipe files for reproducibility and customization.
3. Python APIs for researchers who want to script experiments.

The CLI should support:

```powershell
moe-forge inspect <model>
moe-forge plan <model> --goal balanced
moe-forge run recipe.json
moe-forge eval <dense> <candidate>
```

## Research Standards

Each conversion backend should report:

- source checkpoint identity and architecture
- converted layers and tensor mappings
- expert layout per layer
- router initialization method
- calibration dataset identity and sample count
- recovery training settings
- quality deltas against the dense model
- speed, memory, and active-compute measurements
- known limitations and recommended follow-up runs

## Near-Term Milestones

1. HF model ID intake and local HF checkpoint validation. Initial support exists.
2. Architecture adapters for Llama, Mistral, Qwen, and Gemma-style MLPs. Initial support exists.
3. Tensor mapping registry for gated and ungated FFNs. Initial safetensors/index validation exists.
4. Activation profiler that records neuron frequency, magnitude, and co-activation. Initial hook-based magnitude/frequency profiling exists.
5. Carved-MLP constructor for HF safetensors. Initial validated carve manifest generation and safetensors materialization exist.
6. Router initialization from activation clusters. Initial document-pool router metadata exists.
7. Teacher-KL recovery training.
8. Evaluation reports with JSON and HTML outputs.

## Backend Families

### Carved MLP

Partition dense FFN channels into shared and routed expert groups. This is the first full conversion path because it can be initialized from one dense checkpoint and measured layer by layer.

Current carved artifacts materialize shared/expert tensor slices for inspection and downstream assembly. The next step is writing runnable HF MoE module/config wrappers around those slices.

Current runtime support can verify carved tensors reconstruct source dense FFN weights and can execute all carved experts as a gated MLP parity layer. It can also execute selected expert subsets from document-pool router metadata. The next step is HF wrapper/config generation.

### Sparse Upcycle

Clone or split dense FFNs into experts and continue training until the router and experts specialize. This path needs more compute and should preserve strong experiment tracking.

### Adapter-MoE

Keep the dense model as a shared trunk and route LoRA or adapter experts. This path is useful for laptop experiments, domain specialization, and early router research.

## Open Questions

- Which activation statistics best predict safe channel skipping?
- How should shared channels be selected: frequency, magnitude, clustering centrality, or learned masks?
- Can a router initialized analytically recover enough quality with short training?
- Which layers benefit from MoE conversion on small dense models?
- How much active compute can be removed before instruction behavior degrades?
- Can document-aware expert pools, inspired by Ai2 EMO, produce more reusable expert subsets than global channel-importance carving?
