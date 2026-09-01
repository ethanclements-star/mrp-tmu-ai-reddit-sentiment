import pandas as pd

from reddit_eda.text_analysis import word_frequencies


def dataset_summary(df: pd.DataFrame) -> dict:
    comments_per_post = df.groupby("post_id").size() if "post_id" in df.columns else pd.Series(dtype=int)

    return {
        "comments": len(df),
        "posts": int(df["post_id"].nunique()) if "post_id" in df.columns else 0,
        "authors": int(df["comment_author"].nunique()) if "comment_author" in df.columns else 0,
        "post_start": df["post_created_at"].min() if "post_created_at" in df.columns else None,
        "post_end": df["post_created_at"].max() if "post_created_at" in df.columns else None,
        "comment_start": df["comment_created_at"].min() if "comment_created_at" in df.columns else None,
        "comment_end": df["comment_created_at"].max() if "comment_created_at" in df.columns else None,
        "median_comment_length": float(df["comment_length"].median()) if len(df) else 0.0,
        "mean_comment_length": float(df["comment_length"].mean()) if len(df) else 0.0,
        "median_words": float(df["comment_word_count"].median()) if len(df) else 0.0,
        "median_comments_per_post": float(comments_per_post.median()) if len(comments_per_post) else 0.0,
        "mean_post_score": float(df["post_score"].mean()) if "post_score" in df.columns and len(df) else 0.0,
        "mean_comment_score": float(df["comment_score"].mean()) if "comment_score" in df.columns and len(df) else 0.0,
        "automod_share": float((df["comment_author"] == "AutoModerator").mean()) if "comment_author" in df.columns and len(df) else 0.0,
    }


def monthly_volume(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "comment_created_at" not in df.columns:
        return pd.DataFrame(columns=["month", "comments", "posts"])

    monthly = (
        df.assign(month=df["comment_created_at"].dt.to_period("M").astype(str))
        .groupby("month", as_index=False)
        .agg(comments=("comment_id", "count"), posts=("post_id", "nunique"))
        .sort_values("month")
    )
    return monthly


def length_bins(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["bin", "count"])

    bins = [0, 20, 50, 100, 200, 500, 1000, 10_000]
    labels = ["1-20", "21-50", "51-100", "101-200", "201-500", "501-1000", "1000+"]
    binned = pd.cut(df["comment_length"], bins=bins, labels=labels, right=True, include_lowest=True)
    counts = binned.value_counts().sort_index()
    return pd.DataFrame({"bin": counts.index.astype(str), "count": counts.values})


def top_terms(df: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    return word_frequencies(df, n=n)
