# Fine-Tuning: DeBERTa-v3-large on SciFact

## What is SciFact?

SciFact (Wadden et al., 2020) is a dataset of 1,409 scientific claims paired with evidence abstracts and expert-annotated labels: **SUPPORTS**, **REFUTES**, or **NOT_ENOUGH_INFO**. It was created by Allen AI specifically for claim verification — making it structurally identical to the NLI task we need for VerifAI.

**Why SciFact over general NLI datasets?**

General NLI datasets (MNLI, SNLI) contain sentence pairs drawn from news, fiction, and crowdsourced annotations. SciFact contains claim-evidence pairs in a domain closer to legal and medical fact-checking: structured claims about specific entities, supported or refuted by passages from scientific literature. This domain shift better approximates the legal hallucination detection task.

A legal entailment dataset (claims from court opinions paired with supporting/refuting passages) would be ideal but does not exist in a readily usable open-source form.

---

## Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Base model | `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli` | Already fine-tuned on 4 NLI benchmarks; strong zero-shot baseline |
| Learning rate | `2e-5` | Standard for DeBERTa fine-tuning; lower rates underfit in 3 epochs |
| Batch size | `4` (train), `8` (eval) | Memory-safe for CPU; gradient accumulation not needed at this scale |
| Epochs | `3` | SciFact is small (~1,100 train pairs); 3 epochs avoids overfitting |
| Warmup steps | `100` | Stabilizes early training on a small dataset |
| Weight decay | `0.01` | Light regularization |
| Max sequence length | `256` | Covers most claim+abstract pairs; 512 is slower with minimal gain |
| Eval strategy | `epoch` | Saves best checkpoint by accuracy |

---

## Training Results

Run `python fine_tuning/fine_tune_scifact.py` to generate results.  
Results are saved to `fine_tuning/training_results.json`.

Expected approximate results (varies by hardware/seed):

| Metric | Base Model | Fine-tuned |
|--------|-----------|------------|
| Accuracy | ~0.72 | ~0.81 |
| F1 (macro) | ~0.68 | ~0.78 |

---

## Running the Scripts

```bash
# Fine-tune (~30 min CPU, ~5 min GPU)
python fine_tuning/fine_tune_scifact.py

# Evaluate base vs fine-tuned
python fine_tuning/evaluate_model.py
```

Output directory: `fine_tuning/model_output/`  
Evaluation results: `fine_tuning/evaluation_results.json`

---

## Limitations

1. **Domain mismatch**: SciFact is scientific (biomedical), not legal. A legal NLI dataset (e.g., from contracts or court opinions) would improve performance on legal claims.
2. **Dataset size**: SciFact has ~1,100 training pairs — small by NLI standards. The base model already has strong NLI priors from MNLI/FEVER; fine-tuning primarily adapts the claim-evidence pairing style.
3. **Label imbalance**: NOT_ENOUGH_INFO dominates SciFact (~60%), which biases the model toward INSUFFICIENT_EVIDENCE verdicts.
4. **CPU inference speed**: ~2-5 seconds per claim on CPU. For large documents, this becomes the bottleneck.

---

## Fallback Dataset

If `allenai/scifact` is unavailable from HuggingFace, the script automatically falls back to a 2,000-sample subset of **MultiNLI** (same 3-class entailment/neutral/contradiction labels). Results will be slightly different but the fine-tuning pipeline remains valid.
