"""
Evaluates base vs fine-tuned DeBERTa NLI model on the SciFact validation set.

Computes: accuracy, precision, recall, F1 (macro), confusion matrix.
Saves results to fine_tuning/evaluation_results.json.

Run: python fine_tuning/evaluate_model.py
"""

import os
import sys
import json
import logging
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

BASE_MODEL = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
FINETUNED_PATH = os.path.join(os.path.dirname(__file__), "model_output")
RESULTS_PATH = os.path.join(os.path.dirname(__file__), "evaluation_results.json")

LABEL_NAMES = ["entailment", "neutral", "contradiction"]


def load_val_data():
    """Reuses the same data loading logic from fine_tune_scifact."""
    sys.path.insert(0, os.path.dirname(__file__))
    from fine_tune_scifact import load_scifact_dataset
    _, val_data = load_scifact_dataset()
    return val_data


def evaluate_model(model_name_or_path: str, val_data: list[dict]) -> dict:
    """Runs inference on val_data and returns evaluation metrics."""
    import torch
    import torch.nn.functional as F
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    logger.info("Loading model: %s", model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path)
    model.eval()

    all_preds = []
    all_labels = []

    for item in val_data:
        inputs = tokenizer(
            item["hypothesis"],
            item["premise"],
            truncation=True,
            max_length=256,
            return_tensors="pt",
            padding=True,
        )
        with torch.no_grad():
            logits = model(**inputs).logits
        pred = int(torch.argmax(logits, dim=-1).item())
        all_preds.append(pred)
        all_labels.append(item["label"])

    from sklearn.metrics import (
        accuracy_score,
        precision_recall_fscore_support,
        confusion_matrix,
    )

    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="macro", zero_division=0
    )
    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1, 2]).tolist()

    return {
        "accuracy": round(accuracy, 4),
        "precision_macro": round(precision, 4),
        "recall_macro": round(recall, 4),
        "f1_macro": round(f1, 4),
        "confusion_matrix": cm,
        "confusion_matrix_labels": LABEL_NAMES,
    }


def print_results(name: str, metrics: dict):
    print(f"\n{'=' * 50}")
    print(f"  Model: {name}")
    print(f"{'=' * 50}")
    print(f"  Accuracy:          {metrics['accuracy']:.4f}")
    print(f"  Precision (macro): {metrics['precision_macro']:.4f}")
    print(f"  Recall (macro):    {metrics['recall_macro']:.4f}")
    print(f"  F1 (macro):        {metrics['f1_macro']:.4f}")
    print(f"\n  Confusion Matrix (rows=true, cols=pred):")
    print(f"  Labels: {metrics['confusion_matrix_labels']}")
    for row in metrics["confusion_matrix"]:
        print(f"    {row}")


def main():
    logger.info("Loading validation data...")
    val_data = load_val_data()
    logger.info("Val samples: %d", len(val_data))

    results = {}

    # Evaluate base model
    logger.info("Evaluating base model...")
    base_metrics = evaluate_model(BASE_MODEL, val_data)
    results["base_model"] = {"name": BASE_MODEL, "metrics": base_metrics}
    print_results("Base DeBERTa", base_metrics)

    # Evaluate fine-tuned model if available
    if os.path.isdir(FINETUNED_PATH):
        logger.info("Evaluating fine-tuned model...")
        ft_metrics = evaluate_model(FINETUNED_PATH, val_data)
        results["finetuned_model"] = {"name": FINETUNED_PATH, "metrics": ft_metrics}
        print_results("Fine-tuned DeBERTa (SciFact)", ft_metrics)

        # Delta
        delta_f1 = ft_metrics["f1_macro"] - base_metrics["f1_macro"]
        delta_acc = ft_metrics["accuracy"] - base_metrics["accuracy"]
        print(f"\n  Improvement after fine-tuning:")
        print(f"    F1 delta:       {delta_f1:+.4f}")
        print(f"    Accuracy delta: {delta_acc:+.4f}")
    else:
        logger.warning("Fine-tuned model not found at %s. Run fine_tune_scifact.py first.", FINETUNED_PATH)

    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Results saved to %s", RESULTS_PATH)


if __name__ == "__main__":
    main()
