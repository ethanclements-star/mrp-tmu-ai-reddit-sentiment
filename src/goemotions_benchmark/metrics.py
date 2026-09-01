from dataclasses import dataclass

import pandas as pd

from goemotions_benchmark.labels import (
    EMOTION_NAMES,
    SENTIMENT_CLASSES,
    emotions_to_sentiment,
)


def _safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _lists_to_matrix(lists: list[list[str]], label_names: list[str]) -> list[list[int]]:
    index = {name: idx for idx, name in enumerate(label_names)}
    matrix = []
    for labels in lists:
        row = [0] * len(label_names)
        for label in labels:
            if label in index:
                row[index[label]] = 1
        matrix.append(row)
    return matrix


def _per_label_metrics(
    y_true: list[list[int]],
    y_pred: list[list[int]],
    label_names: list[str],
) -> tuple[pd.DataFrame, list[float]]:
    rows = []
    f1_scores = []

    for idx, name in enumerate(label_names):
        true_vals = [row[idx] for row in y_true]
        pred_vals = [row[idx] for row in y_pred]

        tp = sum(1 for t, p in zip(true_vals, pred_vals) if t == 1 and p == 1)
        fp = sum(1 for t, p in zip(true_vals, pred_vals) if t == 0 and p == 1)
        fn = sum(1 for t, p in zip(true_vals, pred_vals) if t == 1 and p == 0)
        support = sum(true_vals)

        precision = _safe_divide(tp, tp + fp)
        recall = _safe_divide(tp, tp + fn)
        f1 = _safe_divide(2 * precision * recall, precision + recall)
        f1_scores.append(f1)

        rows.append(
            {
                "label": name,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": support,
            }
        )

    return pd.DataFrame(rows), f1_scores


def _micro_f1(y_true: list[list[int]], y_pred: list[list[int]]) -> float:
    tp = fp = fn = 0
    for true_row, pred_row in zip(y_true, y_pred):
        for t, p in zip(true_row, pred_row):
            if t == 1 and p == 1:
                tp += 1
            elif t == 0 and p == 1:
                fp += 1
            elif t == 1 and p == 0:
                fn += 1
    precision = _safe_divide(tp, tp + fp)
    recall = _safe_divide(tp, tp + fn)
    return _safe_divide(2 * precision * recall, precision + recall)


def _subset_accuracy(y_true: list[list[int]], y_pred: list[list[int]]) -> float:
    if not y_true:
        return 0.0
    matches = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    return matches / len(y_true)


def _hamming_loss(y_true: list[list[int]], y_pred: list[list[int]]) -> float:
    if not y_true:
        return 0.0
    total = 0
    wrong = 0
    for true_row, pred_row in zip(y_true, y_pred):
        for t, p in zip(true_row, pred_row):
            total += 1
            if t != p:
                wrong += 1
    return wrong / total


def _jaccard(true: set[str], pred: set[str]) -> float:
    if not true and not pred:
        return 1.0
    union = true | pred
    if not union:
        return 1.0
    return len(true & pred) / len(union)


def _at_least_one_match(true: set[str], pred: set[str]) -> bool:
    if not true and not pred:
        return True
    return bool(true & pred)


def _label_set_overlap_metrics(
    true_labels: list[list[str]],
    pred_labels: list[list[str]],
) -> tuple[float, float]:
    jaccard_scores = []
    match_scores = []
    for true, pred in zip(true_labels, pred_labels):
        true_set = set(true)
        pred_set = set(pred)
        jaccard_scores.append(_jaccard(true_set, pred_set))
        match_scores.append(1.0 if _at_least_one_match(true_set, pred_set) else 0.0)
    n = len(jaccard_scores)
    return sum(jaccard_scores) / n if n else 0.0, sum(match_scores) / n if n else 0.0


@dataclass
class EvaluationResult:
    task: str
    macro_f1: float
    micro_f1: float
    accuracy: float
    hamming_loss: float | None
    per_class: pd.DataFrame
    at_least_one_match: float | None = None
    mean_jaccard: float | None = None


def evaluate_emotions(true_labels: list[list[str]], pred_labels: list[list[str]]) -> EvaluationResult:
    y_true = _lists_to_matrix(true_labels, EMOTION_NAMES)
    y_pred = _lists_to_matrix(pred_labels, EMOTION_NAMES)
    per_class, f1_scores = _per_label_metrics(y_true, y_pred, EMOTION_NAMES)
    mean_jaccard, at_least_one_match = _label_set_overlap_metrics(true_labels, pred_labels)

    return EvaluationResult(
        task="emotions_28",
        macro_f1=sum(f1_scores) / len(f1_scores) if f1_scores else 0.0,
        micro_f1=_micro_f1(y_true, y_pred),
        accuracy=_subset_accuracy(y_true, y_pred),
        hamming_loss=_hamming_loss(y_true, y_pred),
        per_class=per_class,
        at_least_one_match=at_least_one_match,
        mean_jaccard=mean_jaccard,
    )


def evaluate_sentiment(true_labels: list[list[str]], pred_labels: list[list[str]]) -> EvaluationResult:
    true_sentiment = [emotions_to_sentiment(labels) for labels in true_labels]
    pred_sentiment = [emotions_to_sentiment(labels) for labels in pred_labels]

    y_true = _lists_to_matrix([[s] for s in true_sentiment], SENTIMENT_CLASSES)
    y_pred = _lists_to_matrix([[s] for s in pred_sentiment], SENTIMENT_CLASSES)
    per_class, f1_scores = _per_label_metrics(y_true, y_pred, SENTIMENT_CLASSES)

    accuracy = sum(1 for t, p in zip(true_sentiment, pred_sentiment) if t == p) / len(true_sentiment)

    return EvaluationResult(
        task="sentiment_3",
        macro_f1=sum(f1_scores) / len(f1_scores) if f1_scores else 0.0,
        micro_f1=_micro_f1(y_true, y_pred),
        accuracy=accuracy,
        hamming_loss=None,
        per_class=per_class,
    )
