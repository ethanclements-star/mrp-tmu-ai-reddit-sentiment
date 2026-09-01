import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

from goemotions_benchmark.config import DATA_CACHE_DIR, DEFAULT_MODEL_KEYS, MODELS, OUTPUT_DIR
from goemotions_benchmark.data import load_splits
from goemotions_benchmark.metrics import EvaluationResult, evaluate_emotions, evaluate_sentiment
from goemotions_benchmark.models.classical import predict_tfidf_logreg
from goemotions_benchmark.models.llm import DEFAULT_OLLAMA_MODEL, DEFAULT_OLLAMA_URL, predict_ollama
from goemotions_benchmark.models.transformers import predict_transformer
from goemotions_benchmark.plots import (
    plot_macro_f1_comparison,
    plot_per_model_emotion_f1,
    plot_per_model_sentiment_f1,
)


def _result_to_summary_dict(result: EvaluationResult) -> dict:
    summary = {
        "macro_f1": result.macro_f1,
        "micro_f1": result.micro_f1,
        "accuracy": result.accuracy,
        "hamming_loss": result.hamming_loss,
    }
    if result.at_least_one_match is not None:
        summary["at_least_one_match"] = result.at_least_one_match
    if result.mean_jaccard is not None:
        summary["mean_jaccard"] = result.mean_jaccard
    return summary


def _save_result(model_dir: Path, name: str, result: EvaluationResult) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    result.per_class.to_csv(model_dir / f"{name}_per_class.csv", index=False)
    with (model_dir / f"{name}_summary.json").open("w") as f:
        json.dump(_result_to_summary_dict(result), f, indent=2)


def _run_model(
    model_key: str,
    train,
    test,
    ollama_model: str,
    ollama_url: str,
) -> tuple[list[list[str]], float]:
    spec = MODELS[model_key]
    start = time.perf_counter()

    if spec.kind == "classical":
        predictions = predict_tfidf_logreg(train, test)
    elif spec.kind == "llm":
        model_name = spec.ollama_model or ollama_model
        predictions = predict_ollama(test.texts, model=model_name, base_url=ollama_url)
    else:
        predictions = predict_transformer(spec, test.texts)

    elapsed = time.perf_counter() - start
    return predictions, elapsed


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark models on GoEmotions")
    parser.add_argument(
        "--models",
        default=",".join(DEFAULT_MODEL_KEYS),
        help="Comma-separated model keys (default: classical + transformers, not LLM)",
    )
    parser.add_argument(
        "--max-test-samples",
        type=int,
        default=None,
        help="Limit test set size for quick runs",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Base output directory",
    )
    parser.add_argument(
        "--ollama-model",
        default=DEFAULT_OLLAMA_MODEL,
        help=f"Ollama model name (default: {DEFAULT_OLLAMA_MODEL})",
    )
    parser.add_argument(
        "--ollama-url",
        default=DEFAULT_OLLAMA_URL,
        help=f"Ollama API base URL (default: {DEFAULT_OLLAMA_URL})",
    )
    args = parser.parse_args()

    # Keep all HF caches inside the project (avoids permission issues).
    DATA_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    mpl_dir = DATA_CACHE_DIR / "matplotlib"
    mpl_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(mpl_dir)

    model_keys = [key.strip() for key in args.models.split(",") if key.strip()]
    unknown = [key for key in model_keys if key not in MODELS]
    if unknown:
        raise SystemExit(f"Unknown models: {unknown}. Available: {list(MODELS)}")

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.output_dir / run_id
    figures_dir = run_dir / "figures"
    run_dir.mkdir(parents=True, exist_ok=True)

    if any(MODELS[key].kind == "llm" for key in model_keys) and args.max_test_samples is None:
        print(
            "Note: local LLM is slow — use --max-test-samples (e.g. 200) "
            "unless you intend to run the full test set."
        )

    print("Loading GoEmotions...")
    train, test = load_splits(max_test_samples=args.max_test_samples)
    print(f"Train: {len(train.texts)} | Test: {len(test.texts)}")

    summary_rows = []
    emotion_results: dict[str, EvaluationResult] = {}
    sentiment_results: dict[str, EvaluationResult] = {}

    for model_key in model_keys:
        spec = MODELS[model_key]
        print(f"\nRunning {spec.label}...")
        predictions, elapsed = _run_model(
            model_key,
            train,
            test,
            ollama_model=args.ollama_model,
            ollama_url=args.ollama_url,
        )

        emotion_eval = evaluate_emotions(test.emotion_labels, predictions)
        sentiment_eval = evaluate_sentiment(test.emotion_labels, predictions)

        model_dir = run_dir / model_key
        _save_result(model_dir, "emotions", emotion_eval)
        _save_result(model_dir, "sentiment", sentiment_eval)

        emotion_results[model_key] = emotion_eval
        sentiment_results[model_key] = sentiment_eval

        summary_rows.append(
            {
                "model": model_key,
                "model_label": spec.label,
                "runtime_sec": round(elapsed, 2),
                "emotion_macro_f1": round(emotion_eval.macro_f1, 4),
                "emotion_micro_f1": round(emotion_eval.micro_f1, 4),
                "emotion_subset_accuracy": round(emotion_eval.accuracy, 4),
                "emotion_at_least_one_match": round(emotion_eval.at_least_one_match or 0, 4),
                "emotion_mean_jaccard": round(emotion_eval.mean_jaccard or 0, 4),
                "emotion_hamming_loss": round(emotion_eval.hamming_loss or 0, 4),
                "sentiment_macro_f1": round(sentiment_eval.macro_f1, 4),
                "sentiment_micro_f1": round(sentiment_eval.micro_f1, 4),
                "sentiment_accuracy": round(sentiment_eval.accuracy, 4),
            }
        )

        print(
            f"  Emotion macro-F1: {emotion_eval.macro_f1:.4f} | "
            f"Jaccard: {emotion_eval.mean_jaccard:.4f} | "
            f"≥1 label match: {emotion_eval.at_least_one_match:.4f} | "
            f"Runtime: {elapsed:.1f}s"
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(run_dir / "benchmark_summary.csv", index=False)

    plot_macro_f1_comparison(summary_df, figures_dir)
    for model_key in model_keys:
        plot_per_model_emotion_f1(model_key, emotion_results[model_key], figures_dir)
        plot_per_model_sentiment_f1(model_key, sentiment_results[model_key], figures_dir)

    print(f"\nDone. Results saved to: {run_dir}")


if __name__ == "__main__":
    main()
