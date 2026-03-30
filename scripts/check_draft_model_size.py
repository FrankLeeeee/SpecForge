"""
Script to instantiate a draft model from a config.json file and report its parameter count.

Usage:
    python scripts/check_draft_model_size.py --config path/to/config.json [--dtype bfloat16]
"""

import argparse

import torch

from specforge.modeling.auto import AutoDraftModelConfig, AutoEagle3DraftModel


def parse_args():
    parser = argparse.ArgumentParser(
        description="Instantiate a draft model from config.json"
    )
    parser.add_argument("--config", type=str, required=True, help="Path to config.json")
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float32", "float16", "bfloat16"],
        help="Model dtype (default: bfloat16)",
    )
    return parser.parse_args()


def count_parameters(model: torch.nn.Module) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}


def main():
    args = parse_args()

    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    torch_dtype = dtype_map[args.dtype]

    print(f"Loading config from: {args.config}")
    config = AutoDraftModelConfig.from_file(args.config)
    print(f"Config type: {type(config).__name__}")
    print(config)

    print(f"\nInstantiating model with dtype={args.dtype} ...")
    model = AutoEagle3DraftModel.from_config(config, torch_dtype=torch_dtype)
    print(f"Model type: {type(model).__name__}")

    param_counts = count_parameters(model)
    total_m = param_counts["total"] / 1e6
    trainable_m = param_counts["trainable"] / 1e6
    print(f"\nTotal parameters:     {total_m:.2f}M ({param_counts['total']:,})")
    print(f"Trainable parameters: {trainable_m:.2f}M ({param_counts['trainable']:,})")


if __name__ == "__main__":
    main()
