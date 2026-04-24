# VerifAI — AI Hallucination Detection Pipeline

![Python](https://img.shields.io/badge/python-3.11-blue) ![License](https://img.shields.io/badge/license-MIT-green) ![Streamlit](https://img.shields.io/badge/streamlit-deployed-brightgreen)

VerifAI is an AI hallucination detection pipeline that decomposes AI-generated legal text into atomic claims, retrieves evidence from CourtListener's authoritative court opinion database, and scores each claim using DeBERTa-v3 NLI entailment — producing a color-coded transparency report that flags supported, contradicted, and unverifiable claims.

---

## Architecture

```
Input Text
    │
    ▼
[1] Claim Extraction ──── GPT-4o (structured JSON, prompt engineering)
    │
    ▼
[2] Evidence Retrieval ── CourtListener REST API search per claim (RAG)
    │                      + exact case name lookup (hard-check)
    ▼
[3] NLI Scoring ───────── DeBERTa-v3-large (local, CPU)
    │
    ▼
[4] Streamlit Report ──── Color-coded transparency report
```

---

## Features

- **Atomic claim decomposition** via GPT-4o structured JSON output
- **Evidence retrieval from CourtListener** (5M+ court opinions) — RAG without a local corpus
- **Citation existence hard-check** — flags invented case names instantly
- **NLI entailment scoring** with DeBERTa-v3-large
- **Optional fine-tuning** on SciFact dataset for domain-adapted scoring
- **Color-coded Streamlit report** with per-claim verdicts and evidence links

---

## Quick Start

```bash
git clone <repo-url>
cd verifai
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your OpenAI API key
streamlit run app.py
```

> **Note:** First run downloads the NLI model (~1.5 GB from HuggingFace).

---

## Fine-Tuning (Optional)

Fine-tune DeBERTa-v3 on SciFact for domain-adapted claim verification:

```bash
python fine_tuning/fine_tune_scifact.py    # ~30 min on CPU, ~5 min on GPU
python fine_tuning/evaluate_model.py        # Compare base vs fine-tuned
```

Then select **"Fine-tuned DeBERTa"** in the Streamlit sidebar.

See `fine_tuning/README.md` for hyperparameters and results.

---

## Running Tests

```bash
python tests/test_claim_extraction.py   # Stage 1 — requires OPENAI_API_KEY
python tests/test_nli_scoring.py        # Stage 3 — local model only
python tests/test_pipeline_e2e.py       # End-to-end — requires OPENAI_API_KEY
```

---

## Project Structure

```
verifai/
├── app.py                          # Streamlit frontend
├── config.py                       # Env vars, thresholds, model names
├── pipeline.py                     # Pipeline orchestrator (stages 1–3)
├── stages/
│   ├── claim_extraction.py         # Stage 1: GPT-4o claim decomposition
│   ├── evidence_retrieval.py       # Stage 2: CourtListener search + citation hard-check
│   └── nli_scoring.py              # Stage 3: DeBERTa NLI entailment scoring
├── fine_tuning/
│   ├── fine_tune_scifact.py        # Fine-tune DeBERTa on SciFact
│   ├── evaluate_model.py           # Base vs fine-tuned comparison
│   └── README.md                   # Hyperparams, results, limitations
├── tests/
│   ├── test_claim_extraction.py
│   ├── test_nli_scoring.py
│   └── test_pipeline_e2e.py
├── examples/
│   ├── sample_input.txt            # Demo legal brief
│   └── sample_output.json          # Pipeline output for demo text
├── docs/
│   └── index.html                  # GitHub Pages project page
├── requirements.txt
├── .env.example
└── .streamlit/config.toml
```

---

## Technical Details

| Stage | Component | Details |
|-------|-----------|---------|
| Claim Extraction | GPT-4o | Structured JSON, `response_format=json_object`, temperature=0 |
| Evidence Retrieval | CourtListener REST API | `/api/rest/v4/search/?type=o`, top-3 results per claim |
| Citation Hard-Check | CourtListener exact search | Quoted case name query; NOT_FOUND → CONTRADICTED override |
| NLI Scoring | DeBERTa-v3-large | Local CPU inference, ~2–5s per claim |
| Fine-Tuning | SciFact (AllenAI) | 3 epochs, lr=2e-5, batch=4, ~1,100 train pairs |

---

## Evaluation Targets

| Metric | Target |
|--------|--------|
| Hallucination detection recall | > 90% |
| Faithfulness precision | > 85% |
| False positive rate | < 10% |

---

## Ethical Considerations

- VerifAI is a **verification aid**, not a legal oracle. Yellow verdicts (INSUFFICIENT_EVIDENCE) indicate corpus gaps, not confirmed falsity.
- No user data is stored or logged.
- CourtListener data is open-access (Free Law Project).
- The system may produce false positives for claims outside CourtListener's coverage.
- Not designed for adversarial use cases.

---

## Limitations

- **Legal domain only** — medical and general domains coming soon
- **CourtListener coverage** may miss very recent opinions (< 30 days old)
- **NLI model** trained on general-purpose data; fine-tuning on SciFact improves recall but dataset is scientific, not legal
- **Sequential processing** — not optimized for large documents (> 20 claims)

---

## Future Work

- PubMed integration for medical claim verification
- Qdrant vector database for faster offline retrieval
- Domain-specific fine-tuning on legal entailment pairs (e.g., ContractNLI)
- Browser extension for real-time verification
- Batch processing for large documents

---

## Credits

- [CourtListener](https://www.courtlistener.com) — Free Law Project
- [HuggingFace](https://huggingface.co) — Transformers, DeBERTa-v3, SciFact
- [OpenAI](https://openai.com) — GPT-4o API
- [SciFact](https://github.com/allenai/scifact) — Allen Institute for AI
