from __future__ import annotations

import argparse
from pathlib import Path


VOCAB = [
    "<unk>",
    "<pad>",
    "<s>",
    "</s>",
    "moe",
    "forge",
    "routes",
    "carved",
    "experts",
    "through",
    "a",
    "tiny",
    "wrapper",
    "recovery",
    "training",
    "compares",
    "the",
    "student",
    "with",
    "dense",
    "teacher",
    "dataset",
    "backed",
    "smoke",
    "run",
    "should",
    "record",
    "sample",
    "identity",
    "and",
    "artifact",
    "provenance",
    "eval",
    "text",
    "file",
    "token",
    "ids",
    "router",
    "default",
    "pool",
    "all",
    "layer",
    "latency",
    "kl",
    "nll",
    "speed",
    "checkpoint",
    "validation",
    "json",
    "html",
    "config",
    "command",
    "model",
    "local",
    "lab",
    "notebook",
    ".",
    "-",
    "_",
    "1",
    "2",
    "3",
    "4",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a tiny local Llama checkpoint for MoE Forge smoke runs.")
    parser.add_argument("--output", type=Path, default=Path("tiny-llama"), help="Output model directory.")
    parser.add_argument("--seed", type=int, default=1234, help="Torch random seed.")
    parser.add_argument("--skip-tokenizer", action="store_true", help="Write only the model checkpoint.")
    args = parser.parse_args()

    import torch
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

    torch.manual_seed(args.seed)
    config = LlamaConfig(
        attention_bias=False,
        hidden_size=8,
        intermediate_size=16,
        max_position_embeddings=16,
        num_attention_heads=2,
        num_hidden_layers=2,
        num_key_value_heads=2,
        pad_token_id=1,
        bos_token_id=2,
        eos_token_id=3,
        tie_word_embeddings=False,
        vocab_size=len(VOCAB),
    )
    model = LlamaForCausalLM(config)
    args.output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output, safe_serialization=True)
    if not args.skip_tokenizer:
        tokenizer = _toy_tokenizer(PreTrainedTokenizerFast)
        tokenizer.save_pretrained(args.output)
    print(f"wrote {args.output}")


def _toy_tokenizer(tokenizer_cls):
    from tokenizers import Tokenizer, normalizers
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace

    tokenizer = Tokenizer(WordLevel({token: index for index, token in enumerate(VOCAB)}, unk_token="<unk>"))
    tokenizer.normalizer = normalizers.Sequence([normalizers.NFKC(), normalizers.Lowercase()])
    tokenizer.pre_tokenizer = Whitespace()
    return tokenizer_cls(
        tokenizer_object=tokenizer,
        unk_token="<unk>",
        pad_token="<pad>",
        bos_token="<s>",
        eos_token="</s>",
    )


if __name__ == "__main__":
    main()
