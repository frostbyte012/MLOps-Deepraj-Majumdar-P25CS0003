"""
eval.py - Evaluate the fine-tuned DistilBERT model (local or from HuggingFace Hub).

Usage:
    # Evaluate local model:
    python src/eval.py --model_path ./saved_model

    # Evaluate model from HuggingFace Hub:
    python src/eval.py --model_path YOUR_USERNAME/distilbert-goodreads-genre --from_hub
"""

import argparse
import json
import os

import torch
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast, Trainer, TrainingArguments

from data import build_genre_reviews_dict, split_data, build_label_maps, GoodreadsDataset
from utils import compute_metrics, save_evaluation_results, print_classification_report

os.environ["WANDB_DISABLED"] = "true"
MAX_LENGTH = 512


def evaluate(model_path: str, output_path: str = "results/eval_results.json") -> dict:
    """Load model from path/hub, run evaluation, and save results."""

    # 1. Load data
    print("Loading Goodreads data...")
    genre_reviews_dict = build_genre_reviews_dict()
    _, _, test_texts, test_labels = split_data(genre_reviews_dict)

    # 2. Label maps
    train_texts, train_labels, _, _ = split_data(genre_reviews_dict)
    label2id, id2label = build_label_maps(train_labels)

    # 3. Tokenize
    print(f"Loading tokenizer from: {model_path}")
    tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)
    test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=MAX_LENGTH)
    test_labels_enc = [label2id[y] for y in test_labels]
    test_dataset = GoodreadsDataset(test_encodings, test_labels_enc)

    # 4. Load model
    print(f"Loading model from: {model_path}")
    model = DistilBertForSequenceClassification.from_pretrained(model_path)

    # 5. Trainer (eval only – no training args needed beyond output_dir)
    training_args = TrainingArguments(
        output_dir="./eval_tmp",
        per_device_eval_batch_size=16,
        report_to=[],
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )

    # 6. Evaluate
    print("\nRunning evaluation...")
    results = trainer.evaluate()
    print(results)

    # 7. Detailed classification report
    predicted_output = trainer.predict(test_dataset)
    predicted_ids = predicted_output.predictions.argmax(-1).flatten().tolist()
    predicted_labels = [id2label[i] for i in predicted_ids]
    report = print_classification_report(test_labels, predicted_labels)

    results["classification_report"] = report
    results["model_source"] = model_path
    save_evaluation_results(results, output_path)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned DistilBERT")
    parser.add_argument("--model_path", type=str, default="./saved_model",
                        help="Local path or HuggingFace Hub model ID")
    parser.add_argument("--from_hub", action="store_true",
                        help="Set this flag when loading from HuggingFace Hub")
    parser.add_argument("--output", type=str, default="results/eval_results.json")
    args = parser.parse_args()

    results = evaluate(args.model_path, output_path=args.output)
    print(f"\nFinal Accuracy : {results.get('eval_accuracy', 'N/A'):.4f}")
    print(f"Final F1 Score : {results.get('eval_f1_weighted', 'N/A'):.4f}")