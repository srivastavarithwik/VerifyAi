"""
VerifAI — Streamlit frontend for AI hallucination detection.
"""

import os
import json
import streamlit as st
from pipeline import verify_text
from stages.nli_scoring import load_nli_model
from config import (
    BASE_NLI_MODEL,
    FINETUNED_NLI_MODEL_PATH,
    ENTAILMENT_THRESHOLD,
    CONTRADICTION_THRESHOLD,
)

st.set_page_config(page_title="VerifAI", page_icon="⚖️", layout="wide")

DEMO_TEXT = """In the landmark case of Miranda v. Arizona (1966), the Supreme Court established that law enforcement must inform suspects of their constitutional rights before conducting custodial interrogation. This principle was further reinforced in Thompson v. Western Medical Center (2019), where the Ninth Circuit held that patient consent requirements in federal healthcare facilities must follow analogous disclosure standards. The Court in Brown v. Board of Education (1954) unanimously declared that racial segregation in public schools violated the Equal Protection Clause of the Fourteenth Amendment. More recently, in Henderson v. Pacific Healthcare Group (2021), the Seventh Circuit established a three-part test for evaluating informed consent claims in telemedicine settings, requiring documented verbal confirmation, written acknowledgment, and digital timestamp verification. Statistical analysis from the Stanford RegLab found that approximately 33% of AI-generated legal research contains at least one hallucinated citation."""

PAGE_STYLES = """
<style>
/* ── Hide default Streamlit chrome ───────────────────────── */
#MainMenu {visibility: hidden;}
header[data-testid="stHeader"] {display: none;}
.stDeployButton {display: none;}
[data-testid="stSidebarNav"] {display: none;}

/* ── Fixed navbar ────────────────────────────────────────── */
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
    font-size: 1.4rem;
    font-weight: 800;
    color: #fff;
    letter-spacing: -0.5px;
    font-family: sans-serif;
}
.nav-brand span { color: #A5B4FC; }
.nav-links {
    display: flex;
    gap: 2rem;
    align-items: center;
}
.nav-links a {
    color: #C7D2FE;
    text-decoration: none;
    font-size: 0.9rem;
    font-weight: 500;
    font-family: sans-serif;
    transition: color 0.15s;
}
.nav-links a:hover { color: #fff; }
.nav-pill {
    background: rgba(255,255,255,0.14);
    color: #fff;
    padding: 4px 14px;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 700;
    border: 1px solid rgba(255,255,255,0.22);
    font-family: sans-serif;
    letter-spacing: 0.4px;
}

/* Push content below fixed navbar */
.main .block-container { padding-top: 84px !important; }

/* ── Verdict badges ──────────────────────────────────────── */
.badge-supported {
    background: linear-gradient(135deg, #059669, #10B981);
    color: white;
    padding: 3px 12px; border-radius: 12px;
    font-size: 0.82em; font-weight: 700; letter-spacing: 0.3px;
}
.badge-contradicted {
    background: linear-gradient(135deg, #DC2626, #EF4444);
    color: white;
    padding: 3px 12px; border-radius: 12px;
    font-size: 0.82em; font-weight: 700;
}
.badge-insufficient {
    background: linear-gradient(135deg, #D97706, #F59E0B);
    color: white;
    padding: 3px 12px; border-radius: 12px;
    font-size: 0.82em; font-weight: 700;
}

/* ── Evidence passages ───────────────────────────────────── */
.evidence-box {
    background: #EEF2FF;
    border-left: 3px solid #6366F1;
    padding: 10px 14px; margin: 8px 0;
    border-radius: 6px;
    font-size: 0.87em;
    color: #1E293B;
    line-height: 1.55;
}
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


def badge(verdict: str) -> str:
    mapping = {
        "SUPPORTED": "badge-supported",
        "CONTRADICTED": "badge-contradicted",
        "INSUFFICIENT_EVIDENCE": "badge-insufficient",
    }
    label = verdict.replace("_", " ")
    cls = mapping.get(verdict, "badge-insufficient")
    return f'<span class="{cls}">{label}</span>'


@st.cache_resource(show_spinner="Loading NLI model (first run: ~1.5 GB download)...")
def get_model(model_path: str | None):
    return load_nli_model(model_path)


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Settings")

    ent_threshold = st.slider(
        "Entailment threshold", 0.0, 1.0, ENTAILMENT_THRESHOLD, 0.05,
        help="Minimum entailment score to mark a claim as SUPPORTED."
    )
    contra_threshold = st.slider(
        "Contradiction threshold", 0.0, 1.0, CONTRADICTION_THRESHOLD, 0.05,
        help="Contradiction score above which a claim can be CONTRADICTED."
    )

    finetuned_available = os.path.isdir(FINETUNED_NLI_MODEL_PATH)
    model_choice = st.radio(
        "NLI Model",
        options=["Base DeBERTa", "Fine-tuned DeBERTa (SciFact)"],
        index=0,
        help="Fine-tuned option requires running fine_tuning/fine_tune_scifact.py first.",
    )
    if model_choice == "Fine-tuned DeBERTa (SciFact)" and not finetuned_available:
        st.warning("Fine-tuned model not found. Run `python fine_tuning/fine_tune_scifact.py` first.")
        model_choice = "Base DeBERTa"

    st.divider()
    st.subheader("About")
    st.markdown(
        "Built for **INFO 7375 — Generative AI**  \n"
        "Evidence: [CourtListener](https://www.courtlistener.com) (Free Law Project)  \n"
        "NLI Model: DeBERTa-v3-large  \n"
        "[GitHub Repo](https://github.com/placeholder)"
    )
    with st.expander("Technical details"):
        chosen_model = FINETUNED_NLI_MODEL_PATH if (
            model_choice == "Fine-tuned DeBERTa (SciFact)" and finetuned_available
        ) else BASE_NLI_MODEL
        st.code(
            f"Model: {chosen_model}\n"
            f"Entailment threshold: {ent_threshold}\n"
            f"Contradiction threshold: {contra_threshold}\n"
            f"Evidence passages per claim: 3"
        )

# ── Main ─────────────────────────────────────────────────────────────────────
st.title("VerifAI — AI Hallucination Detector")
st.markdown(
    "Paste AI-generated legal text below. VerifAI decomposes it into atomic claims, "
    "retrieves evidence from CourtListener, and scores each claim using NLI entailment."
)

input_text = st.text_area("Input text", value=DEMO_TEXT, height=200)

run_button = st.button("Verify", type="primary", use_container_width=True)

if run_button:
    if not input_text.strip():
        st.error("Please enter some text to verify.")
        st.stop()

    model_path = (
        FINETUNED_NLI_MODEL_PATH
        if (model_choice == "Fine-tuned DeBERTa (SciFact)" and finetuned_available)
        else None
    )
    model, tokenizer = get_model(model_path)

    # Override thresholds from sidebar sliders at runtime
    import config as cfg
    cfg.ENTAILMENT_THRESHOLD = ent_threshold
    cfg.CONTRADICTION_THRESHOLD = contra_threshold

    with st.status("Running VerifAI pipeline...", expanded=True) as status:
        st.write("Extracting claims from text...")
        try:
            from stages.claim_extraction import extract_claims
            claims = extract_claims(input_text)
            st.write(f"Found {len(claims)} claims. Retrieving evidence from CourtListener...")
        except Exception as e:
            st.error(f"Claim extraction failed: {e}")
            st.stop()

        st.write("Scoring claims with NLI model...")
        try:
            result = verify_text(input_text, model=model, tokenizer=tokenizer)
        except Exception as e:
            st.error(f"Pipeline failed: {e}")
            st.stop()

        st.write("Generating report...")
        status.update(label="Done!", state="complete")

    summary = result["summary"]
    claims_data = result["claims"]

    # ── Summary metrics ───────────────────────────────────────────────────────
    st.divider()
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Claims", summary["total_claims"])
    col2.metric("Supported", summary["supported"], delta=f"+{summary['supported']}", delta_color="normal")
    col3.metric("Insufficient Evidence", summary["insufficient"])
    col4.metric("Contradicted", summary["contradicted"], delta=f"-{summary['contradicted']}", delta_color="inverse")
    col5.metric("Trust Score", f"{summary['trust_score']}%")

    st.caption(f"Processed in {result['processing_time_seconds']}s")

    # ── Claim detail cards ────────────────────────────────────────────────────
    st.subheader("Claim-by-Claim Report")

    for c in claims_data:
        truncated = c["claim_text"][:80] + ("…" if len(c["claim_text"]) > 80 else "")
        header_html = f"{badge(c['verdict'])} &nbsp; {truncated}"

        with st.expander(c["claim_id"], expanded=False):
            st.markdown(header_html, unsafe_allow_html=True)
            st.markdown(f"**Claim:** {c['claim_text']}")
            st.info(f"**Source sentence:** {c['source_sentence']}")

            st.progress(c["faithfulness_score"], text=f"Faithfulness score: {c['faithfulness_score']:.2%}")

            citation_icon = {"FOUND": "✅", "NOT_FOUND": "❌", "NOT_APPLICABLE": "➖", "UNAVAILABLE": "⚠️"}
            st.markdown(f"**Citation check:** {citation_icon.get(c['citation_check'], '➖')} {c['citation_check']}")

            if c["citation_override"]:
                st.warning("⚠️ Verdict overridden: citation not found in CourtListener database")

            if c["nli_details"]:
                st.markdown("**Evidence passages:**")
                for ev in c["nli_details"]:
                    st.markdown(
                        f'<div class="evidence-box">'
                        f'{ev["evidence_text"]}'
                        f'<br><small>Entailment: <b>{ev["entailment"]:.2%}</b> &nbsp;|&nbsp; '
                        f'Contradiction: {ev["contradiction"]:.2%} &nbsp;|&nbsp; '
                        f'Neutral: {ev["neutral"]:.2%}</small>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                if c["best_evidence"]["url"]:
                    st.markdown(f"[View best evidence on CourtListener]({c['best_evidence']['url']})")
            else:
                st.caption("No evidence passages retrieved from CourtListener.")

    st.divider()
    st.caption(
        "⚠️ VerifAI is a verification aid, not a substitute for professional judgment. "
        "Yellow claims indicate absence of evidence in the searched corpus, not necessarily falsity."
    )
