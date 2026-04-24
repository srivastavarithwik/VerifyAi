"""
Stage 3: NLI Scoring via fine-tuned DeBERTa-v3

Scores each claim against its retrieved evidence passages using Natural Language
Inference. Determines whether the evidence entails, contradicts, or is neutral
toward each claim.

The model runs locally on CPU — no GPU required, ~2-5 seconds per claim.
"""

import logging
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from config import (
    BASE_NLI_MODEL,
    ENTAILMENT_THRESHOLD,
    CONTRADICTION_THRESHOLD,
    LOW_ENTAILMENT_THRESHOLD,
)

logger = logging.getLogger(__name__)

# Max tokens for NLI input (claim + evidence concatenated)
MAX_LENGTH = 512


def load_nli_model(model_path: str = None):
    """
    Loads the NLI model and tokenizer.

    Args:
        model_path: Path to a fine-tuned model directory. If None or the path
                    doesn't exist, falls back to the base DeBERTa model from HF.

    Returns:
        Tuple of (model, tokenizer). Call once at startup and cache the result.
    """
    import os

    target = model_path if (model_path and os.path.isdir(model_path)) else BASE_NLI_MODEL

    if model_path and not os.path.isdir(model_path):
        logger.warning(
            "Fine-tuned model not found at '%s'. Falling back to base model.", model_path
        )

    logger.info("Loading NLI model: %s", target)
    tokenizer = AutoTokenizer.from_pretrained(target)
    model = AutoModelForSequenceClassification.from_pretrained(target)
    model.eval()
    return model, tokenizer


def _get_label_order(model) -> list[str]:
    """
    Returns the label order from the model's id2label config.
    We do NOT hardcode the order because different checkpoints differ.
    Falls back to a sensible default if config is incomplete.
    """
    id2label = getattr(model.config, "id2label", {})
    if id2label:
        # id2label is {0: "entailment", 1: "neutral", 2: "contradiction"} or similar
        ordered = [id2label[i].lower() for i in sorted(id2label.keys())]
        return ordered
    # Known default for MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli
    return ["entailment", "neutral", "contradiction"]


def _score_pair(claim_text: str, evidence_text: str, model, tokenizer) -> dict:
    """Runs NLI inference on a single (claim, evidence) pair."""
    inputs = tokenizer(
        claim_text,
        evidence_text,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
        padding=True,
    )
    with torch.no_grad():
        logits = model(**inputs).logits

    probs = F.softmax(logits, dim=-1).squeeze().tolist()
    labels = _get_label_order(model)

    result = {label: 0.0 for label in ["entailment", "neutral", "contradiction"]}
    for label, prob in zip(labels, probs):
        if label in result:
            result[label] = round(float(prob), 4)

    return result


def score_claim(claim: dict, evidence: dict, model, tokenizer) -> dict:
    """
    Scores a single claim against its evidence passages using NLI entailment.

    Verdict logic:
      - citation_check == NOT_FOUND  → CONTRADICTED (hard override)
      - faithfulness_score > ENTAILMENT_THRESHOLD  → SUPPORTED
      - max contradiction > CONTRADICTION_THRESHOLD AND low entailment  → CONTRADICTED
      - otherwise  → INSUFFICIENT_EVIDENCE

    Args:
        claim: Claim dict (id, text, type, source_sentence).
        evidence: Evidence dict (claim_id, evidence_passages, citation_check).
        model: Loaded HuggingFace NLI model.
        tokenizer: Corresponding tokenizer.

    Returns:
        Scored claim dict with verdict, scores, and best evidence.
    """
    claim_id = claim["id"]
    claim_text = claim["text"]
    source_sentence = claim.get("source_sentence", "")
    citation_check = evidence.get("citation_check", "NOT_APPLICABLE")
    passages = evidence.get("evidence_passages", [])

    nli_details = []
    best_entailment = 0.0
    best_contradiction = 0.0
    best_evidence = {"text": "", "url": "", "entailment": 0.0}

    for passage in passages:
        evidence_text = passage.get("text", "")
        if not evidence_text or evidence_text == "(no snippet available)":
            continue

        scores = _score_pair(claim_text, evidence_text, model, tokenizer)
        nli_details.append({
            "evidence_text": evidence_text[:300],
            "entailment": scores["entailment"],
            "contradiction": scores["contradiction"],
            "neutral": scores["neutral"],
        })

        if scores["entailment"] > best_entailment:
            best_entailment = scores["entailment"]
            best_evidence = {
                "text": evidence_text[:300],
                "url": passage.get("url", ""),
                "entailment": scores["entailment"],
            }

        if scores["contradiction"] > best_contradiction:
            best_contradiction = scores["contradiction"]

    faithfulness_score = best_entailment

    # Verdict determination
    citation_override = False
    if citation_check == "NOT_FOUND":
        verdict = "CONTRADICTED"
        citation_override = True
    elif faithfulness_score > ENTAILMENT_THRESHOLD:
        verdict = "SUPPORTED"
    elif (
        best_contradiction > CONTRADICTION_THRESHOLD
        and faithfulness_score < LOW_ENTAILMENT_THRESHOLD
    ):
        verdict = "CONTRADICTED"
    else:
        verdict = "INSUFFICIENT_EVIDENCE"

    return {
        "claim_id": claim_id,
        "claim_text": claim_text,
        "source_sentence": source_sentence,
        "verdict": verdict,
        "faithfulness_score": round(faithfulness_score, 4),
        "best_evidence": best_evidence,
        "citation_check": citation_check,
        "citation_override": citation_override,
        "nli_details": nli_details,
    }
