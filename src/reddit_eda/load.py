from pathlib import Path

import pandas as pd

from reddit_eda.config import DEFAULT_DATASET


def load_dataset(path: Path | str = DEFAULT_DATASET) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["comment_body"] = df["comment_body"].fillna("").astype(str)
    df["post_created_at"] = pd.to_datetime(df["post_created_utc"], unit="s", utc=True)
    df["comment_created_at"] = pd.to_datetime(df["comment_created_utc"], unit="s", utc=True)
    df["comment_length"] = df["comment_body"].str.len()
    df["comment_word_count"] = df["comment_body"].str.split().str.len()
    return df
