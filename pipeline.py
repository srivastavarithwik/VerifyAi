"""
Pipeline Orchestrator: chains claim extraction → evidence retrieval → NLI scoring.
"""

import time
import logging
from stages.claim_extraction import extract_claims
from stages.evidence_retrieval import retrieve_evidence
from stages.nli_scoring import score_claim

logger = logging.getLogger(__name__)


def verify_text(text: str, model=None, tokenizer=None) -> dict:
    """
    Runs the full VerifAI pipeline on input text.

    Args:
        text: AI-generated legal text to verify.
        model: Pre-loaded NLI model (pass from Streamlit cache to avoid reloading).
        tokenizer: Corresponding tokenizer.

    Returns:
        {
            "original_text": str,
            "claims": [scored_claim_dict, ...],
            "summary": {
                "total_claims": int,
                "supported": int,
                "contradicted": int,
                "insufficient": int,
                "trust_score": float  # (supported / total) * 100
            },
            "processing_time_seconds": float
        }
    """
    if model is None or tokenizer is None:
        from stages.nli_scoring import load_nli_model
        model, tokenizer = load_nli_model()

    start = time.time()

    # Stage 1: Extract atomic claims
    logger.info("Stage 1: Extracting claims...")
    claims = extract_claims(text)
    logger.info("Extracted %d claims.", len(claims))

    # Stages 2 + 3: Retrieve evidence and score each claim
    scored_claims = []
    for i, claim in enumerate(claims):
        logger.info("Processing claim %d/%d: %s", i + 1, len(claims), claim["id"])

        # Stage 2: Evidence retrieval (includes rate-limit sleep)
        evidence = retrieve_evidence(claim)

        # Stage 3: NLI scoring
        scored = score_claim(claim, evidence, model, tokenizer)
        scored_claims.append(scored)

    # Build summary
    total = len(scored_claims)
    supported = sum(1 for c in scored_claims if c["verdict"] == "SUPPORTED")
    contradicted = sum(1 for c in scored_claims if c["verdict"] == "CONTRADICTED")
    insufficient = sum(1 for c in scored_claims if c["verdict"] == "INSUFFICIENT_EVIDENCE")
    trust_score = round((supported / total * 100) if total > 0 else 0.0, 1)

    elapsed = round(time.time() - start, 2)

    return {
        "original_text": text,
        "claims": scored_claims,
        "summary": {
            "total_claims": total,
            "supported": supported,
            "contradicted": contradicted,
            "insufficient": insufficient,
            "trust_score": trust_score,
        },
        "processing_time_seconds": elapsed,
    }
