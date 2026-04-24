"""
Tests NLI scoring on known claim-evidence pairs with expected verdicts.
Run: python tests/test_nli_scoring.py
No API key required — uses local DeBERTa model only.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from stages.nli_scoring import load_nli_model, _score_pair

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"

THRESHOLD = 0.4  # Score must exceed this to be considered "high"


def run_test(name: str, claim: str, evidence: str, expected_label: str, model, tokenizer):
    print(f"\nTest: {name}")
    scores = _score_pair(claim, evidence, model, tokenizer)
    print(f"  Entailment:    {scores['entailment']:.4f}")
    print(f"  Contradiction: {scores['contradiction']:.4f}")
    print(f"  Neutral:       {scores['neutral']:.4f}")

    winner = max(scores, key=scores.get)
    ok = winner == expected_label or scores[expected_label] > THRESHOLD
    print(f"  Expected: {expected_label} → {PASS if ok else FAIL} (winner: {winner})")
    return ok


def main():
    print("Loading NLI model (may take a moment on first run)...")
    model, tokenizer = load_nli_model()

    results = []

    # Test 1: Clear entailment
    results.append(run_test(
        "Clear entailment (boiling point)",
        claim="Water boils at 100°C at sea level",
        evidence="At standard atmospheric pressure, the boiling point of water is 100 degrees Celsius.",
        expected_label="entailment",
        model=model, tokenizer=tokenizer,
    ))

    # Test 2: Clear contradiction
    results.append(run_test(
        "Clear contradiction (wrong boiling point)",
        claim="Water boils at 50°C at sea level",
        evidence="At standard atmospheric pressure, the boiling point of water is 100 degrees Celsius.",
        expected_label="contradiction",
        model=model, tokenizer=tokenizer,
    ))

    # Test 3: Neutral (unrelated)
    results.append(run_test(
        "Neutral (unrelated topic)",
        claim="The defendant was found liable",
        evidence="The weather in Paris was sunny and warm throughout the weekend.",
        expected_label="neutral",
        model=model, tokenizer=tokenizer,
    ))

    passed = sum(results)
    total = len(results)
    print(f"\n{'=' * 40}")
    print(f"Results: {passed}/{total} passed")
    if passed < total:
        sys.exit(1)


if __name__ == "__main__":
    main()
