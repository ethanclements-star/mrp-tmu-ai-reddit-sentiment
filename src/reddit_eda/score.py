"""Score cleaned Reddit comments with a GoEmotions benchmark model."""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from goemotions_benchmark.config import DATA_CACHE_DIR, DEFAULT_THRESHOLD, MODELS
from goemotions_benchmark.labels import emotions_to_sentiment
from goemotions_benchmark.models.transformers import predict_transformer
from reddit_eda.clean import CleaningOptions, apply_cleaning
from reddit_eda.config import DEFAULT_DATASET, EXPERIMENTS_DIR
from reddit_eda.experiments import (
    METADATA_FILENAME,
    PREDICTIONS_FILENAME,
    metadata_path,
    new_experiment_dir,
    predictions_path,
)
from reddit_eda.load import load_dataset


def _default_cleaning() -> CleaningOptions:
    return CleaningOptions()


def score_comments(
    df: pd.DataFrame,
    model_key: str = "roberta_samlowe",
    batch_size: int = 32,
    threshold: float = DEFAULT_THRESHOLD,
) -> pd.DataFrame:
    if model_key not in MODELS:
        raise ValueError(f"Unknown model: {model_key}. Available: {list(MODELS)}")
    spec = MODELS[model_key]
    if spec.kind != "transformer":
        raise ValueError(f"Model {model_key} is not a transformer model.")

    texts = df["comment_body"].astype(str).tolist()
    predictions = predict_transformer(spec, texts, batch_size=batch_size, threshold=threshold)

    rows = []
    for row, emotion_labels in zip(df.itertuples(index=False), predictions, strict=True):
        sentiment = emotions_to_sentiment(emotion_labels)
        rows.append(
            {
                "comment_id": row.comment_id,
                "post_id": row.post_id,
                "comment_body": row.comment_body,
                "comment_score": row.comment_score,
                "comment_author": row.comment_author,
                "comment_created_at": row.comment_created_at,
                "post_score": getattr(row, "post_score", None),
                "subreddit": getattr(row, "subreddit", None),
                "predicted_emotions": "|".join(emotion_labels),
                "predicted_sentiment": sentiment,
                "model_key": model_key,
                "model_label": spec.label,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score cleaned Reddit comments with a GoEmotions transformer model."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help="Path to Reddit comments CSV",
    )
    parser.add_argument(
        "--model",
        default="roberta_samlowe",
        help="Model key (default: roberta_samlowe)",
    )
    parser.add_argument(
        "--experiment",
        default=None,
        help="Experiment folder name under output/experiments/ (default: reddit_roberta_YYYYMMDD_HHMMSS)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit number of cleaned comments (for quick tests)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Transformer inference batch size",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="Probability threshold for multi-label emotions",
    )
    args = parser.parse_args()

    DATA_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    mpl_dir = DATA_CACHE_DIR / "matplotlib"
    mpl_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(mpl_dir)
    os.environ["HF_HOME"] = str(DATA_CACHE_DIR / "hub")
    os.environ["HF_HUB_CACHE"] = str(DATA_CACHE_DIR / "hub")
    os.environ["TRANSFORMERS_CACHE"] = str(DATA_CACHE_DIR / "hub")
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(DATA_CACHE_DIR / "hub")

    if not args.dataset.exists():
        raise SystemExit(f"Dataset not found: {args.dataset}")

    print(f"Loading {args.dataset} ...")
    raw_df = load_dataset(args.dataset)
    clean_df = apply_cleaning(raw_df, _default_cleaning())
    print(f"Cleaned comments: {len(clean_df):,}")

    if args.max_samples is not None:
        clean_df = clean_df.head(args.max_samples)
        print(f"Using first {len(clean_df):,} comments (--max-samples)")

    if clean_df.empty:
        raise SystemExit("No comments left after cleaning.")

    experiment_dir = new_experiment_dir(args.model, name=args.experiment)
    output_csv = predictions_path(experiment_dir)
    output_meta = metadata_path(experiment_dir)

    spec = MODELS[args.model]
    print(f"Experiment: {experiment_dir.relative_to(EXPERIMENTS_DIR.parent)}")
    print(f"Scoring with {spec.label} ({args.model}) ...")
    start = time.perf_counter()
    predictions_df = score_comments(
        clean_df,
        model_key=args.model,
        batch_size=args.batch_size,
        threshold=args.threshold,
    )
    elapsed = time.perf_counter() - start

    predictions_df.to_csv(output_csv, index=False)

    meta = {
        "experiment": experiment_dir.name,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": str(args.dataset),
        "model_key": args.model,
        "model_label": spec.label,
        "comments_scored": len(predictions_df),
        "runtime_sec": round(elapsed, 2),
        "cleaning": _default_cleaning().__dict__,
        "threshold": args.threshold,
        "predictions_file": PREDICTIONS_FILENAME,
        "metadata_file": METADATA_FILENAME,
    }
    with output_meta.open("w") as f:
        json.dump(meta, f, indent=2)

    print(f"Done in {elapsed:.1f}s")
    print(f"Experiment:  {experiment_dir}")
    print(f"Predictions: {output_csv}")
    print(f"Metadata:    {output_meta}")
    print("\nSentiment breakdown:")
    print(predictions_df["predicted_sentiment"].value_counts().to_string())
    print("\nOpen the Streamlit app → Experiments tab to explore results.")


if __name__ == "__main__":
    main()
