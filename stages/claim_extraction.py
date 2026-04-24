"""
Stage 1: Claim Extraction via GPT-4o

Decomposes AI-generated legal text into atomic, verifiable claims using
structured JSON output. Each claim is independently checkable against a
knowledge source.
"""

import json
from openai import OpenAI
from config import OPENAI_API_KEY, CLAIM_EXTRACTION_MODEL

# System prompt instructs GPT-4o on exactly how to decompose text.
# Each instruction exists for a specific reason — see inline comments.
SYSTEM_PROMPT = """You are a legal fact-checker. Your job is to decompose AI-generated legal text into atomic, independently verifiable claims.

INSTRUCTIONS:

1. ATOMICITY: Split compound sentences into separate atomic claims. Each claim must assert exactly one fact. "X held Y and established Z" becomes two claims.

2. CLAIM TYPES — classify each claim as one of:
   - citation_existence: asserts that a specific legal case exists (e.g., "Thompson v. Western Medical Center (2019) exists")
   - factual: asserts a general fact about law, history, or the world
   - statistical: asserts a number, percentage, or quantitative finding
   - legal_holding: asserts what a specific court decided or ruled in a real case

3. CASE NAME EXTRACTION: For any sentence mentioning a case name, year, court, or legal holding — extract EACH of these as a separate claim. A single sentence like "In Miranda v. Arizona (1966), the Supreme Court held X" yields at minimum:
   - A citation_existence claim: "Miranda v. Arizona (1966) is a real case"
   - A legal_holding claim: "In Miranda v. Arizona (1966), the Supreme Court held X"

4. SOURCE SENTENCE: Preserve the original sentence each claim came from. This enables attribution in the final report.

5. SKIP non-verifiable content:
   - Pure opinions or subjective statements ("this is important", "courts should consider")
   - Predictions or hypotheticals
   - Procedural summaries with no checkable facts
   Do NOT include these as claims.

6. OUTPUT FORMAT: Return a single JSON object with a "claims" array. Each element:
{
  "id": "claim_1",
  "text": "A concise, self-contained statement of the claim",
  "type": "citation_existence | factual | statistical | legal_holding",
  "source_sentence": "The exact original sentence this claim came from"
}

Number claims sequentially: claim_1, claim_2, etc.

EXAMPLE:
Input: "The Supreme Court ruled in Miranda v. Arizona (1966) that suspects must be informed of their rights."
Output:
{
  "claims": [
    {
      "id": "claim_1",
      "text": "Miranda v. Arizona (1966) is a real legal case decided by the Supreme Court",
      "type": "citation_existence",
      "source_sentence": "The Supreme Court ruled in Miranda v. Arizona (1966) that suspects must be informed of their rights."
    },
    {
      "id": "claim_2",
      "text": "In Miranda v. Arizona (1966), the Supreme Court ruled that suspects must be informed of their rights",
      "type": "legal_holding",
      "source_sentence": "The Supreme Court ruled in Miranda v. Arizona (1966) that suspects must be informed of their rights."
    }
  ]
}"""


def extract_claims(text: str) -> list[dict]:
    """
    Decomposes input text into atomic verifiable claims using GPT-4o.

    Args:
        text: AI-generated legal text to analyze.

    Returns:
        List of claim dicts with keys: id, text, type, source_sentence.

    Raises:
        RuntimeError: If the OpenAI API call fails or returns malformed JSON.
        ValueError: If OPENAI_API_KEY is not set.
    """
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY is not set. Check your .env file.")

    client = OpenAI(api_key=OPENAI_API_KEY)

    try:
        response = client.chat.completions.create(
            model=CLAIM_EXTRACTION_MODEL,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Decompose the following text into atomic claims:\n\n{text}"},
            ],
            temperature=0.0,  # Deterministic output for consistent claim extraction
        )
    except Exception as e:
        raise RuntimeError(f"OpenAI API call failed: {e}") from e

    raw = response.choices[0].message.content
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"GPT-4o returned invalid JSON: {e}\nRaw output: {raw}") from e

    claims = parsed.get("claims", [])
    if not isinstance(claims, list):
        raise RuntimeError(f"Expected 'claims' to be a list, got: {type(claims)}")

    # Validate required fields and filter out malformed entries
    valid_claims = []
    for c in claims:
        if all(k in c for k in ("id", "text", "type", "source_sentence")):
            valid_claims.append(c)

    return valid_claims
