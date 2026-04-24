"""
Stage 2: Evidence Retrieval via CourtListener REST API

For every claim, retrieves relevant court opinion snippets from CourtListener.
For citation_existence claims, performs an additional hard-check to verify
whether the named case actually exists in the database.

This is the RAG component: no local corpus, no embeddings — pure API retrieval
from CourtListener's 5M+ court opinion index.
"""

import re
import time
import logging
import requests
from config import (
    COURTLISTENER_API_BASE,
    COURTLISTENER_API_KEY,
    TOP_K_EVIDENCE,
    API_SLEEP_SECONDS,
)

logger = logging.getLogger(__name__)

# Matches case names like "Miranda v. Arizona", "Brown v. Board of Education"
# Stops at a parenthetical year, comma, or period to avoid capturing too much.
CASE_NAME_PATTERN = re.compile(
    r"([A-Z][a-zA-Z'-]+(?:\s+[A-Z][a-zA-Z'-]+)*\s+v\.?\s+[A-Z][a-zA-Z'-]+(?:\s+[A-Z][a-zA-Z'-]+)*)"
)


def _build_headers() -> dict:
    headers = {"Accept": "application/json"}
    if COURTLISTENER_API_KEY:
        headers["Authorization"] = f"Token {COURTLISTENER_API_KEY}"
    return headers


def _search_courtlistener(query: str, top_k: int = TOP_K_EVIDENCE) -> list[dict]:
    """
    Hits the CourtListener search API and returns up to top_k opinion passages.
    Returns an empty list on any network or API error (graceful degradation).
    """
    url = f"{COURTLISTENER_API_BASE}/search/"
    params = {"q": query, "type": "o", "order_by": "score desc"}
    try:
        resp = requests.get(url, params=params, headers=_build_headers(), timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except requests.exceptions.RequestException as e:
        logger.warning("CourtListener search failed for query '%s': %s", query, e)
        return []
    except ValueError as e:
        logger.warning("CourtListener returned non-JSON response: %s", e)
        return []

    results = data.get("results", [])[:top_k]
    passages = []
    for r in results:
        # snippet may be in 'snippet', 'text', or 'caseName' fields
        snippet = r.get("snippet") or r.get("text") or ""
        # Strip HTML tags from snippet
        snippet = re.sub(r"<[^>]+>", " ", snippet).strip()
        passages.append({
            "text": snippet[:800] if snippet else "(no snippet available)",
            "case_name": r.get("caseName") or r.get("case_name") or "Unknown",
            "court": r.get("court") or r.get("court_id") or "Unknown",
            "date": r.get("dateFiled") or r.get("date_filed") or "",
            "url": f"https://www.courtlistener.com{r['absolute_url']}" if r.get("absolute_url") else "",
        })
    return passages


def _extract_case_name(claim_text: str) -> str | None:
    """Extracts the first case name found in a claim string."""
    match = CASE_NAME_PATTERN.search(claim_text)
    return match.group(1).strip() if match else None


def _hard_check_citation(case_name: str) -> str:
    """
    Queries CourtListener for an exact case name.
    Returns 'FOUND', 'NOT_FOUND', or 'UNAVAILABLE'.
    """
    url = f"{COURTLISTENER_API_BASE}/search/"
    params = {"q": f'"{case_name}"', "type": "o"}
    try:
        resp = requests.get(url, params=params, headers=_build_headers(), timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except requests.exceptions.RequestException as e:
        logger.warning("Citation hard-check failed for '%s': %s", case_name, e)
        return "UNAVAILABLE"
    except ValueError:
        return "UNAVAILABLE"

    count = data.get("count", 0)
    results = data.get("results", [])
    if count == 0 and len(results) == 0:
        return "NOT_FOUND"
    return "FOUND"


def retrieve_evidence(claim: dict) -> dict:
    """
    Retrieves evidence passages from CourtListener for a single claim.

    For citation_existence claims, also performs a hard-check to verify
    whether the named case actually exists.

    Args:
        claim: A claim dict with keys: id, text, type, source_sentence.

    Returns:
        Dict with keys:
            claim_id: str
            evidence_passages: list of {text, case_name, court, date, url}
            citation_check: "FOUND" | "NOT_FOUND" | "NOT_APPLICABLE" | "UNAVAILABLE"
    """
    claim_id = claim["id"]
    claim_text = claim["text"]
    claim_type = claim.get("type", "factual")

    # General evidence search for all claim types (this is the RAG step)
    evidence_passages = _search_courtlistener(claim_text)
    time.sleep(API_SLEEP_SECONDS)

    citation_check = "NOT_APPLICABLE"

    if claim_type == "citation_existence":
        case_name = _extract_case_name(claim_text)
        if case_name:
            citation_check = _hard_check_citation(case_name)
            time.sleep(API_SLEEP_SECONDS)
        else:
            # Could not parse a case name — treat as inconclusive
            citation_check = "UNAVAILABLE"

    return {
        "claim_id": claim_id,
        "evidence_passages": evidence_passages,
        "citation_check": citation_check,
    }
