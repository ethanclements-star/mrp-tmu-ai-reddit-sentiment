"""Paths and discovery for experiment result folders."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from reddit_eda.config import EXPERIMENTS_DIR

PREDICTIONS_FILENAME = "predictions.csv"
METADATA_FILENAME = "metadata.json"
FIGURES_DIRNAME = "figures"

MODEL_EXPERIMENT_PREFIX = {
    "roberta_samlowe": "reddit_roberta",
    "bert_monologg": "reddit_bert",
    "tfidf_logreg": "reddit_tfidf",
}


def experiment_prefix(model_key: str) -> str:
    return MODEL_EXPERIMENT_PREFIX.get(model_key, f"reddit_{model_key}")


def new_experiment_dir(model_key: str, name: str | None = None) -> Path:
    """Create `output/experiments/<name>/` with a figures subfolder."""
    if name is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = f"{experiment_prefix(model_key)}_{stamp}"

    path = EXPERIMENTS_DIR / name
    path.mkdir(parents=True, exist_ok=True)
    (path / FIGURES_DIRNAME).mkdir(exist_ok=True)
    return path


def list_experiments(directory: Path = EXPERIMENTS_DIR) -> list[Path]:
    if not directory.exists():
        return []
    runs = [
        path
        for path in directory.iterdir()
        if path.is_dir() and (path / PREDICTIONS_FILENAME).exists()
    ]
    return sorted(runs, key=lambda path: path.stat().st_mtime, reverse=True)


def predictions_path(experiment_dir: Path) -> Path:
    return experiment_dir / PREDICTIONS_FILENAME


def metadata_path(experiment_dir: Path) -> Path:
    return experiment_dir / METADATA_FILENAME


def figures_dir(experiment_dir: Path) -> Path:
    return experiment_dir / FIGURES_DIRNAME
