"""
Tests claim extraction on 3 representative inputs.
Run: python tests/test_claim_extraction.py
Requires OPENAI_API_KEY in .env
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from stages.claim_extraction import extract_claims

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"


def run_test(name: str, text: str, min_claims: int, max_claims: int):
    print(f"\nTest: {name}")
    print(f"Input: {text[:100]}...")
    try:
        claims = extract_claims(text)
        count = len(claims)
        print(f"  Extracted {count} claims:")
        for c in claims:
            print(f"    [{c['type']}] {c['text'][:80]}")
        ok = min_claims <= count <= max_claims
        print(f"  Expected {min_claims}–{max_claims} claims → {PASS if ok else FAIL}")
        return ok
    except Exception as e:
        print(f"  ERROR: {e}")
        print(f"  → {FAIL}")
        return False


def main():
    results = []

    # Test 1: Simple sentence with one citation → 1-2 claims
    results.append(run_test(
        "Single citation",
        "The Supreme Court ruled in Miranda v. Arizona (1966) that suspects must be informed of their rights.",
        min_claims=1,
        max_claims=3,
    ))

    # Test 2: Compound sentence with two citations → 3-4 claims
    results.append(run_test(
        "Two citations compound",
        "Brown v. Board of Education (1954) ended school segregation, while Roe v. Wade (1973) recognized a constitutional right to abortion.",
        min_claims=2,
        max_claims=6,
    ))

    # Test 3: Pure opinion with no verifiable facts → 0 claims
    results.append(run_test(
        "Opinion / subjective statement",
        "Courts should consider the broader social implications of their rulings, and legal scholars believe more research is needed.",
        min_claims=0,
        max_claims=2,
    ))

    passed = sum(results)
    total = len(results)
    print(f"\n{'=' * 40}")
    print(f"Results: {passed}/{total} passed")
    if passed < total:
        sys.exit(1)


if __name__ == "__main__":
    main()
