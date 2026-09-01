from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "output"
DATA_CACHE_DIR = PROJECT_ROOT / "data"

DEFAULT_THRESHOLD = 0.5
RANDOM_SEED = 42


@dataclass(frozen=True)
class ModelSpec:
    key: str
    label: str
    kind: str  # "classical", "transformer", or "llm"
    hf_name: str | None = None
    ollama_model: str | None = None


DEFAULT_MODEL_KEYS = ("tfidf_logreg", "bert_monologg", "roberta_samlowe")

MODELS: dict[str, ModelSpec] = {
    "tfidf_logreg": ModelSpec(
        key="tfidf_logreg",
        label="TF-IDF + Logistic Regression",
        kind="classical",
    ),
    "bert_monologg": ModelSpec(
        key="bert_monologg",
        label="BERT (monologg)",
        kind="transformer",
        hf_name="monologg/bert-base-cased-goemotions-original",
    ),
    "roberta_samlowe": ModelSpec(
        key="roberta_samlowe",
        label="RoBERTa (SamLowe)",
        kind="transformer",
        hf_name="SamLowe/roberta-base-go_emotions",
    ),
    "ollama_llama": ModelSpec(
        key="ollama_llama",
        label="Ollama LLM (local)",
        kind="llm",
        ollama_model="llama3.2:3b",
    ),
}
