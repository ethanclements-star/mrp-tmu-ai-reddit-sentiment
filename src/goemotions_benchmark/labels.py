import json
from importlib import resources

EMOTION_NAMES = [
    "admiration",
    "amusement",
    "anger",
    "annoyance",
    "approval",
    "caring",
    "confusion",
    "curiosity",
    "desire",
    "disappointment",
    "disapproval",
    "disgust",
    "embarrassment",
    "excitement",
    "fear",
    "gratitude",
    "grief",
    "joy",
    "love",
    "nervousness",
    "optimism",
    "pride",
    "realization",
    "relief",
    "remorse",
    "sadness",
    "surprise",
    "neutral",
]

SENTIMENT_CLASSES = ["negative", "neutral", "positive"]
SENTIMENT_PRIORITY = ["negative", "positive", "ambiguous", "neutral"]


def load_sentiment_mapping() -> dict[str, list[str]]:
    with resources.files("goemotions_benchmark").joinpath("sentiment_mapping.json").open() as f:
        return json.load(f)


def emotion_index() -> dict[str, int]:
    return {name: idx for idx, name in enumerate(EMOTION_NAMES)}


def labels_to_emotion_names(labels: list[int]) -> list[str]:
    """GoEmotions simplified config stores active label indices, not a multi-hot vector."""
    return [EMOTION_NAMES[i] for i in labels]


def emotions_to_sentiment_groups(emotions: list[str], mapping: dict[str, list[str]]) -> set[str]:
    groups: set[str] = set()
    for emotion in emotions:
        if emotion == "neutral":
            groups.add("neutral")
            continue
        for group, members in mapping.items():
            if emotion in members:
                groups.add(group)
                break
    return groups


def groups_to_sentiment_label(groups: set[str]) -> str:
    """Collapse Google's groups to a single 3-class label."""
    if not groups:
        return "neutral"
    for group in SENTIMENT_PRIORITY:
        if group in groups:
            if group == "ambiguous":
                return "neutral"
            return group
    return "neutral"


def emotions_to_sentiment(emotions: list[str]) -> str:
    mapping = load_sentiment_mapping()
    groups = emotions_to_sentiment_groups(emotions, mapping)
    return groups_to_sentiment_label(groups)
