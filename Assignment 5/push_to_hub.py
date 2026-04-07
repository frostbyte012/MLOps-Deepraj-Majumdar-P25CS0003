import os
import argparse
from huggingface_hub import HfApi, login

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt',       type=str, required=True)
    parser.add_argument('--repo_name',  type=str, default='frostbyte012/vit-s-lora-cifar100')
    parser.add_argument('--hf_token',   type=str, default=os.getenv('HF_TOKEN', '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'))
    parser.add_argument('--rank',       type=int, default=8)
    parser.add_argument('--alpha',      type=float, default=8)
    parser.add_argument('--dropout',    type=float, default=0.1)
    args = parser.parse_args()

    # Authenticate
    if args.hf_token:
        login(token=args.hf_token, add_to_git_credential=False)
    else:
        login()

    api = HfApi()
    username = api.whoami()['name']
    
    # Fix the double-username bug
    if "/" in args.repo_name:
        repo_id = args.repo_name
    else:
        repo_id = f"{username}/{args.repo_name}"

    # 1. Create the repository
    try:
        api.create_repo(repo_id=repo_id, repo_type='model', exist_ok=True)
    except Exception as e:
        print(f"Repo creation notice: {e}")

    # 2. Upload the model weights
    print(f"Uploading {args.ckpt} to {repo_id}...")
    api.upload_file(
        path_or_fileobj=args.ckpt,
        path_in_repo='best_model.pt',
        repo_id=repo_id,
        repo_type='model',
    )

    # 3. Create and upload the Model Card (using a safe list instead of triple-quotes)
    card_lines = [
        "---",
        "language: en",
        "license: apache-2.0",
        "tags: [image-classification, vit, lora, cifar-100, peft]",
        "datasets: [cifar100]",
        "---",
        "# ViT-Small with LoRA — CIFAR-100",
        "",
        "Fine-tuned ViT-Small on CIFAR-100 using PEFT LoRA.",
        f"- **Rank:** {args.rank}",
        f"- **Alpha:** {args.alpha}",
        f"- **Dropout:** {args.dropout}",
        f"- **Base model:** timm/vit_small_patch16_224",
        ""
    ]
    
    with open('/tmp/README_hf.md', 'w') as f:
        f.write("\n".join(card_lines))
        
    api.upload_file(
        path_or_fileobj='/tmp/README_hf.md',
        path_in_repo='README.md',
        repo_id=repo_id,
        repo_type='model',
    )
    print(f"\n✅ Model pushed to: https://huggingface.co/{repo_id}")

if __name__ == '__main__':
    main()
