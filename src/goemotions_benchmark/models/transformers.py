import os

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from goemotions_benchmark.config import DATA_CACHE_DIR, DEFAULT_THRESHOLD, ModelSpec


def _ensure_cache() -> None:
    cache = DATA_CACHE_DIR / "hub"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(cache)
    os.environ["HF_HUB_CACHE"] = str(cache)
    os.environ["TRANSFORMERS_CACHE"] = str(cache)
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(cache)


def predict_transformer(
    spec: ModelSpec,
    texts: list[str],
    batch_size: int = 32,
    threshold: float = DEFAULT_THRESHOLD,
) -> list[list[str]]:
    _ensure_cache()
    tokenizer = AutoTokenizer.from_pretrained(spec.hf_name)
    model = AutoModelForSequenceClassification.from_pretrained(spec.hf_name)
    model.eval()

    id2label = {int(k): v for k, v in model.config.id2label.items()}
    predictions: list[list[str]] = []

    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start : start + batch_size]
        inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )

        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.sigmoid(logits)

        for row in probs:
            labels = [
                id2label[idx].lower()
                for idx, score in enumerate(row.tolist())
                if score >= threshold
            ]
            predictions.append(labels)

    return predictions
