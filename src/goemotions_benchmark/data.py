import os
from dataclasses import dataclass

from datasets import load_dataset

from goemotions_benchmark.config import DATA_CACHE_DIR, RANDOM_SEED
from goemotions_benchmark.labels import labels_to_emotion_names


@dataclass
class GoEmotionsSplit:
    texts: list[str]
    emotion_labels: list[list[str]]


def _ensure_cache() -> None:
    DATA_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    hub_cache = DATA_CACHE_DIR / "hub"
    hub_cache.mkdir(parents=True, exist_ok=True)
    os.environ["HF_DATASETS_CACHE"] = str(DATA_CACHE_DIR / "datasets")
    os.environ["HF_HOME"] = str(hub_cache)
    os.environ["HF_HUB_CACHE"] = str(hub_cache)
    os.environ["TRANSFORMERS_CACHE"] = str(hub_cache)
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(hub_cache)


def _examples_to_split(examples) -> GoEmotionsSplit:
    texts = []
    emotion_labels = []
    for text, label_row in zip(examples["text"], examples["labels"]):
        texts.append(text.strip())
        emotion_labels.append(labels_to_emotion_names(label_row))
    return GoEmotionsSplit(texts=texts, emotion_labels=emotion_labels)


def load_splits(max_test_samples: int | None = None) -> tuple[GoEmotionsSplit, GoEmotionsSplit]:
    """Load official GoEmotions train and test splits."""
    _ensure_cache()
    dataset = load_dataset("google-research-datasets/go_emotions", "simplified")

    train = _examples_to_split(dataset["train"])
    test = _examples_to_split(dataset["test"])

    if max_test_samples is not None and max_test_samples < len(test.texts):
        rng = __import__("random").Random(RANDOM_SEED)
        indices = list(range(len(test.texts)))
        rng.shuffle(indices)
        indices = sorted(indices[:max_test_samples])
        test = GoEmotionsSplit(
            texts=[test.texts[i] for i in indices],
            emotion_labels=[test.emotion_labels[i] for i in indices],
        )

    return train, test
