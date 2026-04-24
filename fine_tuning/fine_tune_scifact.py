"""
Fine-tune DeBERTa-v3-large on SciFact for domain-specific claim verification.

SciFact maps scientific claims to evidence abstracts with SUPPORTS / REFUTES /
NOT_ENOUGH_INFO labels — the same 3-class NLI structure we use in VerifAI.

If the SciFact dataset is unavailable from HuggingFace, this script falls back
to a 2,000-sample subset of MultiNLI, which has the same label structure.

Run: python fine_tuning/fine_tune_scifact.py
Output: fine_tuning/model_output/
"""

import os
import json
import sys
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

BASE_MODEL = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "model_output")
RESULTS_PATH = os.path.join(os.path.dirname(__file__), "training_results.json")

LABEL_MAP = {
    # SciFact labels
    "SUPPORTS": 0,
    "REFUTES": 2,
    "NOT_ENOUGH_INFO": 1,
    # MultiNLI labels (fallback)
    "entailment": 0,
    "neutral": 1,
    "contradiction": 2,
}
ID2LABEL = {0: "entailment", 1: "neutral", 2: "contradiction"}
LABEL2ID = {v: k for k, v in ID2LABEL.items()}


def load_scifact_dataset():
    """Attempts to load SciFact. Returns (train_data, val_data) as lists of dicts."""
    try:
        from datasets import load_dataset
        logger.info("Attempting to load allenai/scifact from HuggingFace...")
        ds = load_dataset("allenai/scifact", "corpus", trust_remote_code=True)
        # allenai/scifact has a different structure — use the claims split directly
        claims_ds = load_dataset("allenai/scifact", trust_remote_code=True)
        logger.info("SciFact loaded. Splits: %s", list(claims_ds.keys()))

        pairs = []
        for split in ["train", "validation", "test"]:
            if split not in claims_ds:
                continue
            for row in claims_ds[split]:
                label_str = row.get("label") or row.get("verdict")
                if label_str not in LABEL_MAP:
                    continue
                claim = row.get("claim", "")
                evidence = row.get("abstract", "") or row.get("evidence", "")
                if isinstance(evidence, list):
                    evidence = " ".join(str(e) for e in evidence)
                if claim and evidence:
                    pairs.append({
                        "premise": evidence,
                        "hypothesis": claim,
                        "label": LABEL_MAP[label_str],
                    })

        if len(pairs) < 100:
            raise ValueError(f"Too few SciFact pairs parsed: {len(pairs)}")

        split_idx = int(len(pairs) * 0.8)
        return pairs[:split_idx], pairs[split_idx:]

    except Exception as e:
        logger.warning("SciFact load failed (%s). Falling back to MultiNLI subset.", e)
        return load_multinli_fallback()


def load_multinli_fallback():
    """Loads a 2,000-sample subset of MultiNLI as a fallback."""
    from datasets import load_dataset
    logger.info("Loading MultiNLI fallback dataset...")
    ds = load_dataset("multi_nli", split="train[:2000]", trust_remote_code=True)

    label_remap = {"entailment": 0, "neutral": 1, "contradiction": 2}
    pairs = []
    for row in ds:
        label_int = row.get("label", -1)
        if label_int not in (0, 1, 2):
            continue
        pairs.append({
            "premise": row["premise"],
            "hypothesis": row["hypothesis"],
            "label": label_int,
        })

    split_idx = int(len(pairs) * 0.8)
    return pairs[:split_idx], pairs[split_idx:]


def make_hf_dataset(pairs: list[dict]):
    """Converts a list of dicts to a HuggingFace Dataset."""
    from datasets import Dataset
    return Dataset.from_list(pairs)


def tokenize_fn(examples, tokenizer):
    return tokenizer(
        examples["hypothesis"],
        examples["premise"],
        truncation=True,
        max_length=256,
        padding="max_length",
    )


def main():
    from transformers import (
        AutoTokenizer,
        AutoModelForSequenceClassification,
        TrainingArguments,
        Trainer,
    )
    import numpy as np
    from sklearn.metrics import accuracy_score

    logger.info("Loading base model: %s", BASE_MODEL)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL,
        num_labels=3,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,
    )

    train_data, val_data = load_scifact_dataset()
    logger.info("Train size: %d | Val size: %d", len(train_data), len(val_data))

    train_ds = make_hf_dataset(train_data)
    val_ds = make_hf_dataset(val_data)

    train_ds = train_ds.map(lambda x: tokenize_fn(x, tokenizer), batched=True)
    val_ds = val_ds.map(lambda x: tokenize_fn(x, tokenizer), batched=True)

    train_ds.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    val_ds.set_format("torch", columns=["input_ids", "attention_mask", "label"])

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {"accuracy": accuracy_score(labels, preds)}

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=8,
        learning_rate=2e-5,
        warmup_steps=100,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        logging_steps=50,
        fp16=False,  # CPU-safe
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
    )

    logger.info("Starting fine-tuning...")
    train_result = trainer.train()

    logger.info("Saving model to %s", OUTPUT_DIR)
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    # Save training summary
    summary = {
        "base_model": BASE_MODEL,
        "train_samples": len(train_data),
        "val_samples": len(val_data),
        "training_loss": train_result.training_loss,
        "metrics": train_result.metrics,
    }
    with open(RESULTS_PATH, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("Training complete. Results saved to %s", RESULTS_PATH)
    logger.info("Training loss: %.4f", train_result.training_loss)


if __name__ == "__main__":
    main()
