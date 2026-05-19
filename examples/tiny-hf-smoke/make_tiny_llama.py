from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a tiny local Llama checkpoint for MoE Forge smoke runs.")
    parser.add_argument("--output", type=Path, default=Path("tiny-llama"), help="Output model directory.")
    parser.add_argument("--seed", type=int, default=1234, help="Torch random seed.")
    args = parser.parse_args()

    import torch
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(args.seed)
    config = LlamaConfig(
        attention_bias=False,
        hidden_size=8,
        intermediate_size=16,
        max_position_embeddings=16,
        num_attention_heads=2,
        num_hidden_layers=2,
        num_key_value_heads=2,
        tie_word_embeddings=False,
        vocab_size=32,
    )
    model = LlamaForCausalLM(config)
    args.output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output, safe_serialization=True)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
