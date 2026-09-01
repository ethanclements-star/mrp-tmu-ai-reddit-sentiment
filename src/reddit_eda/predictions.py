from pathlib import Path

import pandas as pd

from goemotions_benchmark.labels import EMOTION_NAMES, load_sentiment_mapping
from reddit_eda.experiments import (
    PREDICTIONS_FILENAME,
    list_experiments,
    metadata_path,
    predictions_path,
)

SENTIMENT_ORDER = ["negative", "neutral", "positive"]


def list_prediction_files() -> list[Path]:
    """Return predictions.csv path for each experiment run."""
    return [predictions_path(run) for run in list_experiments()]


def load_predictions(path: Path | str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "predicted_emotions" in df.columns:
        df["predicted_emotions"] = df["predicted_emotions"].fillna("").astype(str)
    if "predicted_sentiment" in df.columns:
        df["predicted_sentiment"] = df["predicted_sentiment"].fillna("neutral").astype(str)
    if "comment_created_at" in df.columns:
        df["comment_created_at"] = pd.to_datetime(df["comment_created_at"], utc=True)
    return df


def load_experiment_metadata(experiment_dir: Path) -> dict | None:
    path = metadata_path(experiment_dir)
    if not path.exists():
        return None
    import json

    with path.open() as f:
        return json.load(f)


def _emotion_lists(df: pd.DataFrame) -> pd.Series:
    return df["predicted_emotions"].map(lambda value: [e for e in value.split("|") if e])


def emotion_frequency_table(df: pd.DataFrame, n: int = 28) -> pd.DataFrame:
    """Comments containing each predicted emotion label at least once."""
    lists = _emotion_lists(df)
    total = len(df)
    rows = []
    for emotion in sorted({label for labels in lists for label in labels}):
        count = int(lists.map(lambda labels: emotion in labels).sum())
        rows.append(
            {
                "emotion": emotion,
                "comments": count,
                "share": round(count / total, 4) if total else 0.0,
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values("comments", ascending=False).head(n).reset_index(drop=True)


def emotion_frequency_all(df: pd.DataFrame) -> pd.DataFrame:
    """All 28 GoEmotions labels, including those with zero predictions."""
    observed = emotion_frequency_table(df, n=28)
    counts = dict(zip(observed["emotion"], observed["comments"]))
    total = len(df)
    rows = []
    for emotion in EMOTION_NAMES:
        count = int(counts.get(emotion, 0))
        rows.append(
            {
                "emotion": emotion,
                "comments": count,
                "share": round(count / total, 4) if total else 0.0,
            }
        )
    return pd.DataFrame(rows).sort_values("comments", ascending=False).reset_index(drop=True)


def emotions_with_sentiment_mapping(df: pd.DataFrame) -> pd.DataFrame:
    """Attach Google's sentiment group (positive / negative / ambiguous / neutral) to each emotion."""
    mapping = load_sentiment_mapping()
    emotion_to_group: dict[str, str] = {"neutral": "neutral"}
    for group, members in mapping.items():
        for emotion in members:
            emotion_to_group[emotion] = "neutral" if group == "ambiguous" else group

    table = emotion_frequency_all(df)
    table["sentiment_group"] = table["emotion"].map(emotion_to_group).fillna("unmapped")
    return table.sort_values(["sentiment_group", "comments"], ascending=[True, False]).reset_index(drop=True)


def monthly_emotion_trend(df: pd.DataFrame, emotions: list[str] | None = None) -> pd.DataFrame:
    if df.empty or "comment_created_at" not in df.columns:
        return pd.DataFrame(columns=["month", "emotion", "comments"])

    working = df.copy()
    working["month"] = working["comment_created_at"].dt.to_period("M").astype(str)
    working["emotion_list"] = _emotion_lists(working)

    if emotions is None:
        top = emotion_frequency_table(df, n=8)["emotion"].tolist()
        emotions = top

    rows = []
    for month, group in working.groupby("month"):
        month_lists = group["emotion_list"]
        for emotion in emotions:
            count = int(month_lists.map(lambda labels: emotion in labels).sum())
            rows.append({"month": month, "emotion": emotion, "comments": count})

    return pd.DataFrame(rows).sort_values(["month", "emotion"])


def monthly_emotion_share(df: pd.DataFrame, emotions: list[str] | None = None) -> pd.DataFrame:
    trend = monthly_emotion_trend(df, emotions=emotions)
    if trend.empty:
        return trend
    totals = trend.groupby("month")["comments"].transform("sum")
    out = trend.copy()
    out["share"] = (out["comments"] / totals).round(4)
    return out


def emotion_cooccurrence(df: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    lists = _emotion_lists(df)
    pair_counts: dict[tuple[str, str], int] = {}
    for labels in lists:
        unique = sorted(set(labels))
        for i, a in enumerate(unique):
            for b in unique[i + 1 :]:
                pair_counts[(a, b)] = pair_counts.get((a, b), 0) + 1

    rows = [
        {"emotion_a": a, "emotion_b": b, "comments": count}
        for (a, b), count in pair_counts.items()
    ]
    if not rows:
        return pd.DataFrame(columns=["pair", "comments"])
    out = pd.DataFrame(rows).sort_values("comments", ascending=False).head(n)
    out["pair"] = out["emotion_a"] + " + " + out["emotion_b"]
    return out.reset_index(drop=True)


def emotion_cooccurrence_matrix(df: pd.DataFrame, top_n: int = 12) -> pd.DataFrame:
    """Symmetric co-occurrence matrix for the top-N most frequent emotions (excluding empty)."""
    top = emotion_frequency_table(df, n=top_n)
    if top.empty:
        return pd.DataFrame()

    emotions = top["emotion"].tolist()
    lists = _emotion_lists(df)
    matrix = pd.DataFrame(0, index=emotions, columns=emotions, dtype=int)

    for labels in lists:
        present = [e for e in emotions if e in labels]
        for i, a in enumerate(present):
            matrix.loc[a, a] += 1
            for b in present[i + 1 :]:
                matrix.loc[a, b] += 1
                matrix.loc[b, a] += 1
    return matrix


def attach_post_scores(predictions: pd.DataFrame, posts_df: pd.DataFrame) -> pd.DataFrame:
    """Join post_score from the source Reddit dataset onto prediction rows."""
    if "post_score" in predictions.columns:
        return predictions
    if "post_id" not in predictions.columns or "post_id" not in posts_df.columns:
        return predictions
    if "post_score" not in posts_df.columns:
        return predictions

    post_scores = (
        posts_df[["post_id", "post_score"]]
        .drop_duplicates(subset="post_id")
    )
    return predictions.merge(post_scores, on="post_id", how="left")


def avg_post_score_by_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    if "post_score" not in df.columns:
        return pd.DataFrame()

    grouped = (
        df.groupby("predicted_sentiment", as_index=False)
        .agg(
            mean_post_score=("post_score", "mean"),
            median_post_score=("post_score", "median"),
            comments=("post_score", "count"),
        )
    )
    order = {label: idx for idx, label in enumerate(SENTIMENT_ORDER)}
    grouped["sort_key"] = grouped["predicted_sentiment"].map(order)
    return grouped.sort_values("sort_key").drop(columns="sort_key").reset_index(drop=True)


def sentiment_distribution(df: pd.DataFrame) -> pd.DataFrame:
    counts = df["predicted_sentiment"].value_counts()
    total = len(df)
    rows = []
    for label in SENTIMENT_ORDER:
        count = int(counts.get(label, 0))
        rows.append(
            {
                "sentiment": label,
                "comments": count,
                "share": round(count / total, 4) if total else 0.0,
            }
        )
    return pd.DataFrame(rows)


def monthly_sentiment_trend(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "comment_created_at" not in df.columns:
        return pd.DataFrame(columns=["month", "sentiment", "comments"])

    working = df.copy()
    working["month"] = working["comment_created_at"].dt.to_period("M").astype(str)
    grouped = (
        working.groupby(["month", "predicted_sentiment"], as_index=False)
        .size()
        .rename(columns={"size": "comments"})
    )
    return grouped.sort_values("month")


def monthly_sentiment_share(df: pd.DataFrame) -> pd.DataFrame:
    trend = monthly_sentiment_trend(df)
    if trend.empty:
        return trend

    totals = trend.groupby("month")["comments"].transform("sum")
    trend = trend.copy()
    trend["share"] = (trend["comments"] / totals).round(4)
    return trend


def sentiment_by_score_bucket(df: pd.DataFrame) -> pd.DataFrame:
    if "comment_score" not in df.columns:
        return pd.DataFrame()

    working = df.copy()
    working["score_bucket"] = pd.cut(
        working["comment_score"],
        bins=[-100, -1, 0, 1, 5, 20, 1000],
        labels=["<-1", "-1 to 0", "1", "2-5", "6-20", "20+"],
    )
    return (
        working.groupby(["score_bucket", "predicted_sentiment"], observed=True)
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )


def avg_score_by_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    if "comment_score" not in df.columns:
        return pd.DataFrame()

    grouped = (
        df.groupby("predicted_sentiment", as_index=False)
        .agg(mean_score=("comment_score", "mean"), comments=("comment_score", "count"))
    )
    order = {label: idx for idx, label in enumerate(SENTIMENT_ORDER)}
    grouped["sort_key"] = grouped["predicted_sentiment"].map(order)
    return grouped.sort_values("sort_key").drop(columns="sort_key").reset_index(drop=True)


def emotion_count_per_comment(df: pd.DataFrame) -> pd.DataFrame:
    counts = df["predicted_emotions"].str.split("|").map(len)
    distribution = counts.value_counts().sort_index()
    return pd.DataFrame({"labels_per_comment": distribution.index, "comments": distribution.values})


def filter_predictions(df: pd.DataFrame, sentiment: str | None = None, emotion: str | None = None) -> pd.DataFrame:
    out = df
    if sentiment and sentiment != "All":
        out = out[out["predicted_sentiment"] == sentiment]
    if emotion and emotion != "All":
        out = out[out["predicted_emotions"].str.split("|").map(lambda labels: emotion in labels)]
    return out
