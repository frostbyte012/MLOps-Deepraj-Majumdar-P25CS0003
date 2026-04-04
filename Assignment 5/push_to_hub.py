"""
Assignment 5 — Push best LoRA ViT-S checkpoint to HuggingFace Hub.

Usage:
    python push_to_hub.py \
        --ckpt ./results/Q1/lora_r8_a8/lora_r8_a8_best.pt \
        --repo <your-username>/vit-s-lora-cifar100

The script:
  1. Rebuilds the same LoRA model used during training (r=8, α=8 by default).
  2. Loads the best checkpoint weights.
  3. Pushes the PEFT adapter + base model config to HuggingFace Hub.
"""

import argparse
import os
import torch
from transformers import ViTForImageClassification, ViTConfig
from peft import LoraConfig, get_peft_model, TaskType
from huggingface_hub import login


def build_lora_vit(rank: int, alpha: float, dropout: float = 0.1,
                   num_classes: int = 100):
    config = ViTConfig.from_pretrained('WinKawaks/vit-small-patch16-224')
    config.num_labels = num_classes
    model = ViTForImageClassification.from_pretrained(
        'WinKawaks/vit-small-patch16-224',
        config=config,
        ignore_mismatched_sizes=True,
    )
    lora_cfg = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=rank,
        lora_alpha=alpha,
        target_modules=["query", "key", "value"],
        lora_dropout=dropout,
        bias="none",
        modules_to_save=["classifier"],
    )
    model = get_peft_model(model, lora_cfg)
    return model


def main():
    parser = argparse.ArgumentParser(description='Push best LoRA ViT-S to HuggingFace Hub')
    parser.add_argument('--ckpt',     type=str, required=True,
                        help='Path to best .pt checkpoint (state_dict)')
    parser.add_argument('--repo',     type=str, default='<your-username>/vit-s-lora-cifar100',
                        help='HuggingFace repo id, e.g. johndoe/vit-s-lora-cifar100')
    parser.add_argument('--rank',     type=int,   default=8)
    parser.add_argument('--alpha',    type=float, default=8)
    parser.add_argument('--dropout',  type=float, default=0.1)
    parser.add_argument('--hf_token', type=str,   default=None,
                        help='HuggingFace write token (or set HF_TOKEN env var)')
    args = parser.parse_args()

    # ── Authenticate ───────────────────────────────────────────────────────
    token = args.hf_token or os.environ.get('HF_TOKEN')
    if token:
        login(token=token)
    else:
        # Tries cached credentials from `huggingface-cli login`
        print("No --hf_token provided; using cached HuggingFace credentials.")

    # ── Rebuild model & load weights ───────────────────────────────────────
    print(f"Building LoRA ViT-S (r={args.rank}, α={args.alpha}) ...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_lora_vit(args.rank, args.alpha, args.dropout)

    print(f"Loading checkpoint: {args.ckpt}")
    state = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state)
    model.eval()

    # ── Push to Hub ────────────────────────────────────────────────────────
    print(f"Pushing to HuggingFace Hub: {args.repo} ...")
    model.push_to_hub(args.repo)
    print(f"✅ Done! Model available at https://huggingface.co/{args.repo}")


if __name__ == '__main__':
    main()