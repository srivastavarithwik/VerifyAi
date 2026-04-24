"""
End-to-end pipeline test using the demo legal brief.

Assertions:
  Miranda v. Arizona    → SUPPORTED or INSUFFICIENT_EVIDENCE (real case)
  Brown v. Board        → SUPPORTED or INSUFFICIENT_EVIDENCE (real case)
  Thompson v. Western   → CONTRADICTED (fake case — citation not found)
  Henderson v. Pacific  → CONTRADICTED (fake case — citation not found)

Saves output to examples/sample_output.json.
Run: python tests/test_pipeline_e2e.py
Requires OPENAI_API_KEY in .env
"""

import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from pipeline import verify_text
from stages.nli_scoring import load_nli_model

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"

DEMO_TEXT = """In the landmark case of Miranda v. Arizona (1966), the Supreme Court established that law enforcement must inform suspects of their constitutional rights before conducting custodial interrogation. This principle was further reinforced in Thompson v. Western Medical Center (2019), where the Ninth Circuit held that patient consent requirements in federal healthcare facilities must follow analogous disclosure standards. The Court in Brown v. Board of Education (1954) unanimously declared that racial segregation in public schools violated the Equal Protection Clause of the Fourteenth Amendment. More recently, in Henderson v. Pacific Healthcare Group (2021), the Seventh Circuit established a three-part test for evaluating informed consent claims in telemedicine settings, requiring documented verbal confirmation, written acknowledgment, and digital timestamp verification. Statistical analysis from the Stanford RegLab found that approximately 33% of AI-generated legal research contains at least one hallucinated citation."""

EXAMPLES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "examples")
OUTPUT_PATH = os.path.join(EXAMPLES_DIR, "sample_output.json")


def find_claim_verdict(claims: list[dict], keyword: str) -> str | None:
    """Finds the verdict for the claim whose text contains keyword (case-insensitive)."""
    keyword_lower = keyword.lower()
    for c in claims:
        if keyword_lower in c["claim_text"].lower():
            return c["verdict"]
    return None


def main():
    print("Loading NLI model...")
    model, tokenizer = load_nli_model()

    print("\nRunning full pipeline on demo text...")
    result = verify_text(DEMO_TEXT, model=model, tokenizer=tokenizer)

    summary = result["summary"]
    claims = result["claims"]

    print(f"\nSummary:")
    print(f"  Total claims:   {summary['total_claims']}")
    print(f"  Supported:      {summary['supported']}")
    print(f"  Contradicted:   {summary['contradicted']}")
    print(f"  Insufficient:   {summary['insufficient']}")
    print(f"  Trust score:    {summary['trust_score']}%")
    print(f"  Processing:     {result['processing_time_seconds']}s")

    print("\n--- Claim verdicts ---")
    for c in claims:
        override = " [CITATION OVERRIDE]" if c.get("citation_override") else ""
        print(f"  {c['claim_id']}: {c['verdict']}{override}")
        print(f"    {c['claim_text'][:90]}")

    results_ok = []

    # Assertion 1: Miranda v. Arizona → real case, should not be CONTRADICTED via citation
    miranda_verdict = find_claim_verdict(claims, "Miranda v. Arizona")
    if miranda_verdict:
        ok = miranda_verdict in ("SUPPORTED", "INSUFFICIENT_EVIDENCE")
        print(f"\nMiranda v. Arizona verdict: {miranda_verdict} → {PASS if ok else FAIL}")
        results_ok.append(ok)
    else:
        print(f"\nMiranda v. Arizona: not found in extracted claims → {FAIL}")
        results_ok.append(False)

    # Assertion 2: Brown v. Board → real case
    brown_verdict = find_claim_verdict(claims, "Brown v. Board")
    if brown_verdict:
        ok = brown_verdict in ("SUPPORTED", "INSUFFICIENT_EVIDENCE")
        print(f"Brown v. Board verdict: {brown_verdict} → {PASS if ok else FAIL}")
        results_ok.append(ok)
    else:
        print(f"Brown v. Board: not found in extracted claims → {FAIL}")
        results_ok.append(False)

    # Assertion 3: Thompson v. Western Medical Center → fake, should be CONTRADICTED
    thompson_verdict = find_claim_verdict(claims, "Thompson v. Western")
    if thompson_verdict:
        ok = thompson_verdict == "CONTRADICTED"
        print(f"Thompson v. Western verdict: {thompson_verdict} → {PASS if ok else FAIL}")
        results_ok.append(ok)
    else:
        print(f"Thompson v. Western: not found in extracted claims → {FAIL}")
        results_ok.append(False)

    # Assertion 4: Henderson v. Pacific Healthcare → fake, should be CONTRADICTED
    henderson_verdict = find_claim_verdict(claims, "Henderson v. Pacific")
    if henderson_verdict:
        ok = henderson_verdict == "CONTRADICTED"
        print(f"Henderson v. Pacific verdict: {henderson_verdict} → {PASS if ok else FAIL}")
        results_ok.append(ok)
    else:
        print(f"Henderson v. Pacific: not found in extracted claims → {FAIL}")
        results_ok.append(False)

    # Save output
    os.makedirs(EXAMPLES_DIR, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\nOutput saved to {OUTPUT_PATH}")

    passed = sum(results_ok)
    total = len(results_ok)
    print(f"\n{'=' * 40}")
    print(f"Results: {passed}/{total} passed")

    if passed < total:
        sys.exit(1)


if __name__ == "__main__":
    main()
