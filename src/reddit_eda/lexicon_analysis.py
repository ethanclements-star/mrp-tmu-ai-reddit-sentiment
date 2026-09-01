import re

import pandas as pd

# Generative-art / tool terms relevant to the visual-art Reddit sample.
TRACKED_KEYWORDS: list[tuple[str, str]] = [
    ("midjourney", r"\bmidjourney\b"),
    ("dall-e", r"\bdall[\s-]?e\b"),
    ("stable diffusion", r"\bstable diffusion\b"),
    ("generated", r"\bgenerated\b"),
    ("artificial", r"\bartificial\b"),
    ("deepfake", r"\bdeepfake\b"),
    ("prompt", r"\bprompt\b"),
    ("sora", r"\bsora\b"),
    ("fake", r"\bfake\b"),
    ("real", r"\breal\b"),
    ("photorealistic", r"\bphotorealistic\b"),
]

POSITIVE_WORDS = [
    "love", "amazing", "beautiful", "great", "awesome", "incredible", "stunning",
    "wow", "perfect", "excellent", "good", "nice", "best", "fantastic", "brilliant",
    "impressive", "wonderful", "happy", "cool", "thanks", "thank", "gorgeous",
    "masterpiece", "favorite", "favourite", "enjoy", "liked", "helpful",
]

NEGATIVE_WORDS = [
    "hate", "awful", "terrible", "bad", "ugly", "worse", "worst", "garbage",
    "trash", "stupid", "disappointing", "boring", "creepy", "unsettling", "wrong",
    "horrible", "disgusting", "annoying", "fake", "soulless", "dead", "lazy",
    "bland", "cheap", "scam", "fear", "worried", "concern", "problem", "issue",
]


def _count_matches(pattern: str, text: str) -> int:
    return len(re.findall(pattern, text, flags=re.IGNORECASE))


def keyword_frequency_table(
    df: pd.DataFrame,
    keywords: list[tuple[str, str]] | None = None,
) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(
            columns=["keyword", "comments_with_keyword", "total_mentions", "pct_comments"]
        )

    keywords = keywords or TRACKED_KEYWORDS
    total_comments = len(df)
    rows = []

    for label, pattern in keywords:
        mentions = 0
        comments_with = 0
        for text in df["comment_body"].astype(str):
            count = _count_matches(pattern, text)
            if count:
                comments_with += 1
                mentions += count
        rows.append(
            {
                "keyword": label,
                "comments_with_keyword": comments_with,
                "total_mentions": mentions,
                "pct_comments": round(comments_with / total_comments * 100, 2),
            }
        )

    return pd.DataFrame(rows).sort_values("comments_with_keyword", ascending=False)


def emotional_word_counts(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["word", "polarity", "count"])

    counts: dict[tuple[str, str], int] = {}
    for word in POSITIVE_WORDS:
        counts[(word, "positive")] = 0
    for word in NEGATIVE_WORDS:
        counts[(word, "negative")] = 0

    for text in df["comment_body"].astype(str):
        lowered = text.lower()
        for word in POSITIVE_WORDS:
            counts[(word, "positive")] += len(re.findall(rf"\b{re.escape(word)}\b", lowered))
        for word in NEGATIVE_WORDS:
            counts[(word, "negative")] += len(re.findall(rf"\b{re.escape(word)}\b", lowered))

    rows = [
        {"word": word, "polarity": polarity, "count": count}
        for (word, polarity), count in counts.items()
        if count > 0
    ]
    result = pd.DataFrame(rows)
    if result.empty:
        return pd.DataFrame(columns=["word", "polarity", "count"])
    return result.sort_values("count", ascending=False).reset_index(drop=True)


def emotional_summary(df: pd.DataFrame) -> dict:
    if df.empty:
        return {
            "positive_mentions": 0,
            "negative_mentions": 0,
            "comments_with_positive": 0,
            "comments_with_negative": 0,
            "pct_positive_comments": 0.0,
            "pct_negative_comments": 0.0,
        }

    pos_pattern = re.compile(
        r"\b(" + "|".join(re.escape(w) for w in POSITIVE_WORDS) + r")\b",
        re.IGNORECASE,
    )
    neg_pattern = re.compile(
        r"\b(" + "|".join(re.escape(w) for w in NEGATIVE_WORDS) + r")\b",
        re.IGNORECASE,
    )

    pos_comments = neg_comments = 0
    pos_mentions = neg_mentions = 0

    for text in df["comment_body"].astype(str):
        pos_hits = pos_pattern.findall(text)
        neg_hits = neg_pattern.findall(text)
        if pos_hits:
            pos_comments += 1
            pos_mentions += len(pos_hits)
        if neg_hits:
            neg_comments += 1
            neg_mentions += len(neg_hits)

    total = len(df)
    return {
        "positive_mentions": pos_mentions,
        "negative_mentions": neg_mentions,
        "comments_with_positive": pos_comments,
        "comments_with_negative": neg_comments,
        "pct_positive_comments": round(pos_comments / total * 100, 2),
        "pct_negative_comments": round(neg_comments / total * 100, 2),
    }
