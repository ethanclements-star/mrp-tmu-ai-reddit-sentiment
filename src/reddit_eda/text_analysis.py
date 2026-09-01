import re
from collections import Counter

import pandas as pd

# Function words, contractions, and a few Reddit discourse fillers.
# Apostrophes are stripped before lookup, so "it's" matches "its".
STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "also", "am", "an",
    "and", "any", "are", "arent", "as", "at", "be", "because", "been", "before",
    "being", "below", "between", "both", "but", "by", "can", "cant", "could",
    "couldnt", "did", "didnt", "do", "does", "doesnt", "doing", "dont", "down",
    "during", "each", "even", "few", "for", "from", "further", "get", "gonna",
    "gotta", "had", "hadnt", "has", "hasnt", "have", "havent", "having", "he",
    "her", "here", "hers", "herself", "him", "himself", "his", "how", "i", "if",
    "im", "in", "into", "is", "isnt", "it", "its", "itself", "just", "like", "ll",
    "me", "more", "most", "much", "my", "myself", "no", "nor", "not", "now", "of",
    "off", "ok", "okay", "on", "once", "one", "only", "or", "other", "our", "ours",
    "ourselves", "out", "over", "own", "re", "really", "s", "same", "she", "should",
    "shouldnt", "so", "some", "such", "t", "than", "that", "thats", "the", "their",
    "theirs", "them", "themselves", "then", "there", "theres", "these", "they",
    "theyre", "this", "those", "through", "to", "too", "under", "until", "up", "us",
    "ve", "very", "wanna", "was", "wasnt", "we", "were", "werent", "what", "whats",
    "when", "where", "which", "while", "who", "whom", "why", "will", "with", "wont",
    "would", "wouldnt", "y", "yeah", "you", "youd", "youll", "your", "youre",
    "yours", "yourself", "yourselves", "ive", "youve", "weve", "theyve", "theyll",
    "theyd", "aint", "lets", "heres", "whos", "still",
}

TOKEN_PATTERN = re.compile(r"[a-z0-9']+")
EMOJI_PATTERN = re.compile(
    "["
    "\U0001F600-\U0001F64F"
    "\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F6FF"
    "\U0001F1E0-\U0001F1FF"
    "\U00002702-\U000027B0"
    "\U000024C2-\U0001F251"
    "]+",
    flags=re.UNICODE,
)


def _normalize_token(token: str, min_length: int) -> str | None:
    clean = token.lower().replace("'", "")
    if len(clean) < min_length or clean in STOPWORDS:
        return None
    return clean


def tokenize_series(texts: pd.Series, min_length: int = 3) -> list[str]:
    tokens: list[str] = []
    for text in texts.astype(str):
        for raw in TOKEN_PATTERN.findall(text.lower()):
            token = _normalize_token(raw, min_length)
            if token:
                tokens.append(token)
    return tokens


def word_frequencies(df: pd.DataFrame, n: int = 50, min_length: int = 3) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["term", "count", "share"])

    counts = Counter(tokenize_series(df["comment_body"], min_length=min_length))
    total = sum(counts.values()) or 1
    rows = [{"term": term, "count": count, "share": count / total} for term, count in counts.most_common(n)]
    return pd.DataFrame(rows)


def bigram_frequencies(df: pd.DataFrame, n: int = 30, min_length: int = 3) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["bigram", "count"])

    bigrams: Counter[str] = Counter()
    for text in df["comment_body"].astype(str):
        tokens = [
            t
            for raw in TOKEN_PATTERN.findall(text.lower())
            if (t := _normalize_token(raw, min_length))
        ]
        bigrams.update(f"{tokens[i]} {tokens[i + 1]}" for i in range(len(tokens) - 1))

    rows = [{"bigram": bg, "count": count} for bg, count in bigrams.most_common(n)]
    return pd.DataFrame(rows)


def emoji_frequencies(df: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["emoji", "count"])

    counts: Counter[str] = Counter()
    for text in df["comment_body"].astype(str):
        for match in EMOJI_PATTERN.findall(text):
            counts[match] += 1

    rows = [{"emoji": emoji, "count": count} for emoji, count in counts.most_common(n)]
    return pd.DataFrame(rows)


def comments_containing_term(df: pd.DataFrame, term: str, limit: int = 10) -> tuple[pd.DataFrame, int]:
    if df.empty or not term:
        return pd.DataFrame(), 0

    pattern = re.compile(rf"\b{re.escape(term.lower())}\b")
    mask = df["comment_body"].astype(str).str.lower().map(lambda t: bool(pattern.search(t)))
    matches = df.loc[mask, ["post_title", "comment_body", "comment_score", "comment_author"]]
    return matches.head(limit).reset_index(drop=True), len(matches)


def weekday_volume(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "comment_created_at" not in df.columns:
        return pd.DataFrame(columns=["weekday", "comments"])

    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    counts = (
        df.assign(weekday=df["comment_created_at"].dt.day_name())
        .groupby("weekday", as_index=False)
        .agg(comments=("comment_id", "count"))
    )
    counts["weekday"] = pd.Categorical(counts["weekday"], categories=order, ordered=True)
    return counts.sort_values("weekday")


def top_authors(df: pd.DataFrame, n: int = 15) -> pd.DataFrame:
    if df.empty or "comment_author" not in df.columns:
        return pd.DataFrame(columns=["comment_author", "comments", "mean_score"])

    return (
        df.groupby("comment_author", as_index=False)
        .agg(comments=("comment_id", "count"), mean_score=("comment_score", "mean"))
        .sort_values("comments", ascending=False)
        .head(n)
    )
