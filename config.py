import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CLAIM_EXTRACTION_MODEL = "gpt-4o"

# NLI Models
BASE_NLI_MODEL = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
FINETUNED_NLI_MODEL_PATH = "fine_tuning/model_output"

# CourtListener
COURTLISTENER_API_BASE = "https://www.courtlistener.com/api/rest/v4"
COURTLISTENER_API_KEY = os.getenv("COURTLISTENER_API_KEY", "")

# Scoring Thresholds
ENTAILMENT_THRESHOLD = 0.7
CONTRADICTION_THRESHOLD = 0.5
LOW_ENTAILMENT_THRESHOLD = 0.3
TOP_K_EVIDENCE = 3
API_SLEEP_SECONDS = 0.5
