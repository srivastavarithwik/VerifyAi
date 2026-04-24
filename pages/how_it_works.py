"""
VerifAI — How It Works: architecture and pipeline details.
"""

import streamlit as st

st.set_page_config(page_title="VerifAI — How It Works", page_icon="⚖️", layout="wide")

PAGE_STYLES = """
<style>
#MainMenu {visibility: hidden;}
header[data-testid="stHeader"] {display: none;}
.stDeployButton {display: none;}

.verifai-navbar {
    position: fixed;
    top: 0; left: 0; right: 0;
    z-index: 999;
    background: linear-gradient(135deg, #1E1B4B 0%, #3730A3 100%);
    height: 58px;
    padding: 0 2.5rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    box-shadow: 0 2px 16px rgba(0,0,0,0.28);
}
.nav-brand {
    font-size: 1.4rem; font-weight: 800; color: #fff;
    letter-spacing: -0.5px; font-family: sans-serif;
}
.nav-brand span { color: #A5B4FC; }
.nav-links { display: flex; gap: 2rem; align-items: center; }
.nav-links a {
    color: #C7D2FE; text-decoration: none;
    font-size: 0.9rem; font-weight: 500; font-family: sans-serif;
}
.nav-links a:hover { color: #fff; }
.nav-pill {
    background: rgba(255,255,255,0.14); color: #fff;
    padding: 4px 14px; border-radius: 20px;
    font-size: 0.78rem; font-weight: 700;
    border: 1px solid rgba(255,255,255,0.22);
    font-family: sans-serif; letter-spacing: 0.4px;
}
.main .block-container { padding-top: 84px !important; }

/* Pipeline flow diagram */
.pipeline-wrap {
    display: flex; align-items: stretch;
    gap: 0; margin: 2rem 0; overflow-x: auto;
}
.pipe-stage {
    flex: 1; min-width: 160px;
    background: linear-gradient(160deg, #312E81, #4338CA);
    color: white; border-radius: 12px;
    padding: 20px 16px; text-align: center;
    position: relative;
}
.pipe-stage h4 { margin: 0 0 6px 0; font-size: 0.95rem; font-weight: 700; color: #A5B4FC; }
.pipe-stage p { margin: 0; font-size: 0.8rem; color: #E0E7FF; line-height: 1.45; }
.pipe-arrow {
    display: flex; align-items: center; padding: 0 6px;
    font-size: 1.4rem; color: #6366F1; font-weight: 700; flex-shrink: 0;
}
.pipe-input {
    background: linear-gradient(160deg, #0F172A, #1E293B);
}
.pipe-output {
    background: linear-gradient(160deg, #064E3B, #065F46);
}

/* Detail cards */
.arch-card {
    background: white;
    border: 1px solid #E0E7FF;
    border-radius: 12px;
    padding: 24px 28px;
    margin-bottom: 16px;
    box-shadow: 0 1px 6px rgba(79,70,229,0.07);
}
.arch-card h3 {
    margin: 0 0 12px 0; color: #312E81;
    font-size: 1.1rem; display: flex; align-items: center; gap: 10px;
}
.stage-badge {
    background: #4F46E5; color: white;
    padding: 2px 10px; border-radius: 20px;
    font-size: 0.72rem; font-weight: 700; letter-spacing: 0.5px;
}
.code-pill {
    background: #EEF2FF; color: #3730A3;
    padding: 2px 8px; border-radius: 6px;
    font-family: monospace; font-size: 0.82em;
}
.verdict-table { width: 100%; border-collapse: collapse; margin-top: 10px; font-size: 0.88rem; }
.verdict-table th {
    background: #312E81; color: white;
    padding: 8px 14px; text-align: left; font-weight: 600;
}
.verdict-table td { padding: 8px 14px; border-bottom: 1px solid #E0E7FF; }
.verdict-table tr:hover td { background: #F5F3FF; }
.chip-supported { background: #D1FAE5; color: #065F46; padding: 2px 10px; border-radius: 10px; font-weight: 700; font-size: 0.8em; }
.chip-contradicted { background: #FEE2E2; color: #991B1B; padding: 2px 10px; border-radius: 10px; font-weight: 700; font-size: 0.8em; }
.chip-insufficient { background: #FEF3C7; color: #92400E; padding: 2px 10px; border-radius: 10px; font-weight: 700; font-size: 0.8em; }
</style>

<div class="verifai-navbar">
    <div class="nav-brand">Verif<span>AI</span></div>
    <div class="nav-links">
        <a href="/">Home</a>
        <a href="/how_it_works">How It Works</a>
        <a href="#">About</a>
        <span class="nav-pill">INFO 7375</span>
    </div>
</div>
"""

st.markdown(PAGE_STYLES, unsafe_allow_html=True)

st.title("How VerifAI Works")
st.markdown("A three-stage pipeline that decomposes AI-generated legal text into verifiable claims, retrieves real court opinions as evidence, and scores each claim using NLI entailment.")

st.divider()

# ── Pipeline diagram ──────────────────────────────────────────────────────────
st.subheader("Pipeline Overview")

st.markdown("""
<div class="pipeline-wrap">
    <div class="pipe-stage pipe-input">
        <h4>Input</h4>
        <p>AI-generated legal text (paragraphs, memos, summaries)</p>
    </div>
    <div class="pipe-arrow">→</div>
    <div class="pipe-stage">
        <h4>Stage 1</h4>
        <p>Claim Extraction<br><small>GPT-4o · JSON output</small></p>
    </div>
    <div class="pipe-arrow">→</div>
    <div class="pipe-stage">
        <h4>Stage 2</h4>
        <p>Evidence Retrieval<br><small>CourtListener API · RAG</small></p>
    </div>
    <div class="pipe-arrow">→</div>
    <div class="pipe-stage">
        <h4>Stage 3</h4>
        <p>NLI Scoring<br><small>DeBERTa-v3-large · local CPU</small></p>
    </div>
    <div class="pipe-arrow">→</div>
    <div class="pipe-stage pipe-output">
        <h4>Output</h4>
        <p>Per-claim verdicts + trust score</p>
    </div>
</div>
""", unsafe_allow_html=True)

st.divider()

# ── Stage 1 ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="arch-card">
<h3><span class="stage-badge">STAGE 1</span> Claim Extraction &nbsp;·&nbsp; <span class="code-pill">stages/claim_extraction.py</span></h3>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns([3, 2])
with col1:
    st.markdown("**What it does**")
    st.markdown("""
GPT-4o receives the full input text and a detailed system prompt instructing it to decompose the text into **atomic, independently verifiable claims** — one fact per claim, no compound statements.

Each claim is classified into one of four types:

| Type | Description |
|---|---|
| `citation_existence` | Asserts a specific legal case exists (e.g. *Thompson v. Western Medical Center*) |
| `legal_holding` | Asserts what a court decided in a real case |
| `factual` | Asserts a general fact about law or history |
| `statistical` | Asserts a number, percentage, or quantitative finding |

The output is **structured JSON**, enforced by OpenAI's `response_format: json_object`. Temperature is set to `0.0` for deterministic, consistent extraction.
""")

with col2:
    st.markdown("**Example decomposition**")
    st.code("""Input sentence:
"In Miranda v. Arizona (1966), the
Supreme Court held that suspects
must be informed of their rights."

→ claim_1 (citation_existence):
  "Miranda v. Arizona (1966) is a
   real legal case"

→ claim_2 (legal_holding):
  "In Miranda v. Arizona (1966),
   the Supreme Court held that
   suspects must be informed of
   their rights"
""", language="text")
    st.markdown("**Key design choice**")
    st.info("A single sentence containing a case name always yields at minimum two claims — one to verify the citation exists, one to verify the ruling. This catches hallucinations at two independent levels.")

st.divider()

# ── Stage 2 ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="arch-card">
<h3><span class="stage-badge">STAGE 2</span> Evidence Retrieval (RAG) &nbsp;·&nbsp; <span class="code-pill">stages/evidence_retrieval.py</span></h3>
</div>
""", unsafe_allow_html=True)

col3, col4 = st.columns([3, 2])
with col3:
    st.markdown("**What it does**")
    st.markdown("""
For each claim, the pipeline queries the **CourtListener REST API** — a free, open database of 5M+ US court opinions maintained by the Free Law Project.

**Two retrieval operations run per claim:**

1. **General semantic search** — the claim text is sent as a keyword query to CourtListener's search endpoint (`/search/?type=o`). The top 3 opinion snippets are returned and used as evidence passages in Stage 3. This is the RAG step — no local corpus, no embeddings, no vector database.

2. **Citation hard-check** (only for `citation_existence` claims) — the extracted case name is searched with exact quotes (e.g. `"Thompson v. Western Medical Center"`). If CourtListener returns zero results, the case is flagged `NOT_FOUND` and the verdict is immediately set to `CONTRADICTED`, bypassing NLI scoring entirely.
""")
    st.markdown("**Rate limiting**")
    st.markdown("A configurable `API_SLEEP_SECONDS` pause is inserted between requests to avoid hitting CourtListener's rate limits.")

with col4:
    st.markdown("**Citation hard-check logic**")
    st.code("""# Exact-quoted search on case name
params = {
  "q": '"Thompson v. Western Medical"',
  "type": "o"
}
resp = requests.get(
  COURTLISTENER_API_BASE + "/search/",
  params=params
)
count = resp.json().get("count", 0)

if count == 0:
    return "NOT_FOUND"   # hallucination
else:
    return "FOUND"       # case exists
""", language="python")
    st.markdown("**Return values**")
    st.markdown("""
| Value | Meaning |
|---|---|
| `FOUND` | Case exists in CourtListener |
| `NOT_FOUND` | No match — likely hallucinated |
| `NOT_APPLICABLE` | Non-citation claim type |
| `UNAVAILABLE` | API error or unparseable name |
""")

st.divider()

# ── Stage 3 ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="arch-card">
<h3><span class="stage-badge">STAGE 3</span> NLI Scoring &nbsp;·&nbsp; <span class="code-pill">stages/nli_scoring.py</span></h3>
</div>
""", unsafe_allow_html=True)

col5, col6 = st.columns([3, 2])
with col5:
    st.markdown("**What it does**")
    st.markdown("""
**Natural Language Inference (NLI)** asks: *does this evidence support, contradict, or say nothing about this claim?*

The model used is **`MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli`** — a DeBERTa-v3-large checkpoint fine-tuned on five NLI datasets (MultiNLI, FEVER, ANLI, Ling, WANLI). It runs fully locally on CPU with no GPU required.

Each `(claim, evidence_passage)` pair is tokenized and fed through the model. The softmax output gives three probabilities:

- **Entailment** — evidence supports the claim
- **Contradiction** — evidence refutes the claim
- **Neutral** — evidence is unrelated or inconclusive

The **faithfulness score** for a claim equals the highest entailment score across all its retrieved passages.
""")

    st.markdown("**Verdict decision logic**")
    st.markdown("""
<table class="verdict-table">
<tr><th>Condition (evaluated in order)</th><th>Verdict</th></tr>
<tr><td>Citation hard-check returned <code>NOT_FOUND</code></td><td><span class="chip-contradicted">CONTRADICTED</span> (hard override)</td></tr>
<tr><td>Best entailment score &gt; entailment threshold (default 0.50)</td><td><span class="chip-supported">SUPPORTED</span></td></tr>
<tr><td>Best contradiction &gt; contradiction threshold (default 0.30) AND entailment &lt; low threshold (0.20)</td><td><span class="chip-contradicted">CONTRADICTED</span></td></tr>
<tr><td>All other cases</td><td><span class="chip-insufficient">INSUFFICIENT EVIDENCE</span></td></tr>
</table>
""", unsafe_allow_html=True)

with col6:
    st.markdown("**Model input format**")
    st.code("""# Tokenizer receives claim + evidence
# as a sentence pair (NLI standard)
inputs = tokenizer(
    claim_text,       # hypothesis
    evidence_text,    # premise
    truncation=True,
    max_length=512,
    return_tensors="pt",
)

logits = model(**inputs).logits
probs = softmax(logits)
# → [entailment, neutral, contradiction]
""", language="python")

    st.markdown("**Why DeBERTa?**")
    st.info("DeBERTa-v3 uses disentangled attention — it encodes token content and position separately — which significantly outperforms BERT and RoBERTa on NLI benchmarks. The large variant scores ~92% on MultiNLI.")

    st.markdown("**Fine-tuning (optional)**")
    st.markdown("A fine-tuned checkpoint on the **SciFact** dataset can be loaded instead of the base model via the sidebar. SciFact is a scientific claim verification dataset — training on it improves sensitivity to factual precision. Run `python fine_tuning/fine_tune_scifact.py` to produce the checkpoint.")

st.divider()

# ── Tech stack ────────────────────────────────────────────────────────────────
st.subheader("Tech Stack")
c1, c2, c3 = st.columns(3)
with c1:
    st.markdown("**AI / Models**")
    st.markdown("""
- GPT-4o (OpenAI API) — claim extraction
- DeBERTa-v3-large (HuggingFace) — NLI scoring
- SciFact fine-tune (optional) — improved factual precision
""")
with c2:
    st.markdown("**Retrieval**")
    st.markdown("""
- CourtListener REST API — 5M+ US court opinions
- No vector database
- No local corpus or embeddings
- Pure keyword + exact-match API retrieval
""")
with c3:
    st.markdown("**Infrastructure**")
    st.markdown("""
- Streamlit — frontend
- PyTorch + Transformers — NLI inference (CPU)
- python-dotenv — API key management
- `datasets` + `scikit-learn` — fine-tuning utilities
""")

st.divider()
st.caption("Built for INFO 7375 — Generative AI · Northeastern University")
