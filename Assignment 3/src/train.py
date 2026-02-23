"""
train.py - Fine-tune DistilBERT for Goodreads genre classification.

Usage:
    python src/train.py [--epochs 3] [--batch_size 10] [--hf_token YOUR_TOKEN]
                        [--push_to_hub] [--hub_model_id YOUR_USERNAME/model-name]
"""

import argparse
import os

from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizerFast,
    Trainer,
    TrainingArguments,
)

from data import build_genre_reviews_dict, split_data, build_label_maps, GoodreadsDataset
from utils import compute_metrics, save_evaluation_results, print_classification_report, set_seed

# ── Constants ─────────────────────────────────────────────────────────────────
MODEL_NAME = "distilbert-base-cased"
MAX_LENGTH = 512
CACHED_MODEL_DIR = "./saved_model"
os.environ["WANDB_DISABLED"] = "true"


def run_baseline(train_texts, train_labels, test_texts, test_labels) -> dict:
    """TF-IDF + Logistic Regression baseline."""
    print("\n--- Baseline: TF-IDF + Logistic Regression ---")
    vectorizer = TfidfVectorizer()
    X_train = vectorizer.fit_transform(train_texts)
    X_test = vectorizer.transform(test_texts)

    model = LogisticRegression(max_iter=1000).fit(X_train, train_labels)
    predictions = model.predict(X_test)
    print_classification_report(test_labels, predictions)
    return {"baseline_predictions": predictions.tolist()}


def train(args):
    set_seed(42)

    # 1. Load data
    print("Loading Goodreads data...")
    genre_reviews_dict = build_genre_reviews_dict()
    train_texts, train_labels, test_texts, test_labels = split_data(genre_reviews_dict)
    print(f"Train: {len(train_texts)} | Test: {len(test_texts)}")

    # 2. Baseline
    run_baseline(train_texts, train_labels, test_texts, test_labels)

    # 3. Build label maps
    label2id, id2label = build_label_maps(train_labels)
    num_labels = len(label2id)

    # 4. Tokenize
    print(f"\nTokenizing with {MODEL_NAME}...")
    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)
    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=MAX_LENGTH)
    test_encodings  = tokenizer(test_texts,  truncation=True, padding=True, max_length=MAX_LENGTH)

    train_labels_enc = [label2id[y] for y in train_labels]
    test_labels_enc  = [label2id[y] for y in test_labels]

    train_dataset = GoodreadsDataset(train_encodings, train_labels_enc)
    test_dataset  = GoodreadsDataset(test_encodings,  test_labels_enc)

    # 5. Load model
    print(f"\nLoading pre-trained model: {MODEL_NAME}")
    model = DistilBertForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
    )

    # 6. Training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=16,
        learning_rate=5e-5,
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=100,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        report_to=[],
    )

    # 7. Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )

    # 8. Train
    print("\nStarting fine-tuning...")
    trainer.train()

    # 9. Save locally
    print(f"\nSaving model to {CACHED_MODEL_DIR}")
    trainer.save_model(CACHED_MODEL_DIR)
    tokenizer.save_pretrained(CACHED_MODEL_DIR)

    # 10. Evaluate
    print("\nRunning final evaluation...")
    eval_results = trainer.evaluate()
    print(eval_results)
    save_evaluation_results(eval_results, "results/local_eval_results.json")

    # 11. (Optional) Push to Hugging Face Hub
    if args.push_to_hub:
        if not args.hf_token:
            raise ValueError("--hf_token required when using --push_to_hub")
        from huggingface_hub import login, HfApi
        login(token=args.hf_token)

        print(f"\nPushing model to HuggingFace Hub: {args.hub_model_id}")

        # Push model and tokenizer directly (more reliable than trainer.push_to_hub)
        model.push_to_hub(args.hub_model_id, token=args.hf_token)
        tokenizer.push_to_hub(args.hub_model_id, token=args.hf_token)

        # Verify upload succeeded before eval step tries to load it
        api = HfApi()
        files = list(api.list_repo_files(args.hub_model_id))
        print(f"Files on Hub: {files}")
        has_weights = any(f in files for f in ["model.safetensors", "pytorch_model.bin"])
        if not has_weights:
            raise RuntimeError(
                f"Model weights missing on Hub after push! Files found: {files}"
            )
        print(f"Model successfully verified at: https://huggingface.co/{args.hub_model_id}")

    return trainer, tokenizer, id2label, test_texts, test_labels, test_dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune DistilBERT on Goodreads genre data")
    parser.add_argument("--epochs",       type=int,  default=3)
    parser.add_argument("--batch_size",   type=int,  default=10)
    parser.add_argument("--push_to_hub",  action="store_true")
    parser.add_argument("--hf_token",     type=str,  default=None)
    parser.add_argument("--hub_model_id", type=str,  default="YOUR_USERNAME/distilbert-goodreads-genre")
    args = parser.parse_args()

    train(args)