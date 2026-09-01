from dataclasses import dataclass

import pandas as pd

from reddit_eda.config import BOT_AUTHORS, DELETED_PATTERNS
from reddit_eda.text_utils import strip_urls


@dataclass
class CleaningOptions:
    remove_automoderator: bool = True
    remove_deleted: bool = True
    strip_urls: bool = True
    remove_empty: bool = True
    min_comment_length: int = 0
    dedupe_comments: bool = True


def _is_deleted(text: str) -> bool:
    stripped = text.strip().lower()
    return stripped in {p.lower() for p in DELETED_PATTERNS}


def _refresh_text_metrics(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["comment_length"] = out["comment_body"].str.len()
    out["comment_word_count"] = out["comment_body"].str.split().str.len()
    return out


def _apply_steps(df: pd.DataFrame, options: CleaningOptions, track_steps: bool) -> tuple[pd.DataFrame, list[tuple[str, int]]]:
    steps: list[tuple[str, int]] = [("Raw dataset", len(df))]
    working = df

    if options.dedupe_comments and "comment_id" in working.columns:
        working = working.drop_duplicates(subset="comment_id", keep="first")
        steps.append(("After deduplicating comments", len(working)))

    if options.remove_automoderator and "comment_author" in working.columns:
        working = working[~working["comment_author"].isin(BOT_AUTHORS)]
        steps.append(("After removing AutoModerator", len(working)))

    if options.remove_deleted:
        working = working[~working["comment_body"].map(_is_deleted)]
        steps.append(("After removing [deleted] / [removed]", len(working)))

    if options.strip_urls:
        working = working.copy()
        working["comment_body"] = working["comment_body"].astype(str).map(strip_urls)
        working = _refresh_text_metrics(working)
        steps.append(("After stripping URLs", len(working)))

    if options.remove_empty:
        working = working[working["comment_body"].str.strip() != ""]
        steps.append(("After removing empty comments", len(working)))

    if options.min_comment_length > 0:
        working = working[working["comment_length"] >= options.min_comment_length]
        steps.append((f"After min length ({options.min_comment_length})", len(working)))

    steps.append(("Final cleaned dataset", len(working)))
    if not track_steps:
        steps = []
    return working.reset_index(drop=True), steps


def cleaning_funnel(df: pd.DataFrame, options: CleaningOptions) -> pd.DataFrame:
    _, steps = _apply_steps(df, options, track_steps=True)
    return pd.DataFrame(steps, columns=["step", "rows"])


def apply_cleaning(df: pd.DataFrame, options: CleaningOptions) -> pd.DataFrame:
    cleaned, _ = _apply_steps(df, options, track_steps=False)
    return cleaned
