from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from goemotions_benchmark.metrics import EvaluationResult


def _save_figure(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_macro_f1_comparison(summary_df: pd.DataFrame, figures_dir: Path) -> None:
    models = summary_df["model"].tolist()
    x = range(len(models))
    width = 0.35

    plt.figure(figsize=(10, 5))
    plt.bar([i - width / 2 for i in x], summary_df["emotion_macro_f1"], width, label="28 emotions")
    plt.bar([i + width / 2 for i in x], summary_df["sentiment_macro_f1"], width, label="3 sentiment")
    plt.xticks(list(x), models, rotation=20, ha="right")
    plt.ylabel("Macro F1")
    plt.title("Macro F1 comparison across models")
    plt.legend()
    plt.ylim(0, 1)
    _save_figure(figures_dir / "benchmark_macro_f1_comparison.png")


def plot_per_model_emotion_f1(model_key: str, result: EvaluationResult, figures_dir: Path) -> None:
    df = result.per_class.sort_values("f1", ascending=True)

    plt.figure(figsize=(8, 10))
    plt.barh(df["label"], df["f1"], color="steelblue")
    plt.xlabel("F1 score")
    plt.title(f"Per-emotion F1: {model_key}")
    plt.xlim(0, 1)
    _save_figure(figures_dir / f"{model_key}_emotion_f1_by_class.png")


def plot_per_model_sentiment_f1(model_key: str, result: EvaluationResult, figures_dir: Path) -> None:
    df = result.per_class.sort_values("f1", ascending=True)

    plt.figure(figsize=(6, 4))
    plt.barh(df["label"], df["f1"], color="darkorange")
    plt.xlabel("F1 score")
    plt.title(f"Per-sentiment F1 (mapped): {model_key}")
    plt.xlim(0, 1)
    _save_figure(figures_dir / f"{model_key}_sentiment_f1_by_class.png")
