from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from wordcloud import WordCloud

from reddit_eda.text_analysis import word_frequencies

# Report-ready defaults: white background, dark text.
plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.edgecolor": "none",
        "text.color": "black",
        "axes.labelcolor": "black",
        "xtick.color": "black",
        "ytick.color": "black",
        "axes.edgecolor": "#333333",
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.color": "#cccccc",
    }
)


def prepare_figure(fig: plt.Figure) -> plt.Figure:
    """Force white backgrounds for on-screen display and PNG export."""
    fig.patch.set_facecolor("white")
    for ax in fig.axes:
        ax.set_facecolor("white")
    return fig


def fig_to_png_bytes(fig: plt.Figure) -> bytes:
    prepare_figure(fig)
    buffer = BytesIO()
    fig.savefig(
        buffer,
        format="png",
        dpi=200,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
        transparent=False,
    )
    plt.close(fig)
    buffer.seek(0)
    return buffer.getvalue()


def comment_length_histogram(df: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 4), facecolor="white")
    ax.set_facecolor("white")
    ax.hist(df["comment_length"], bins=40, color="steelblue", edgecolor="white")
    ax.set_xlabel("Comment length (characters)")
    ax.set_ylabel("Count")
    ax.set_title("Comment length distribution")
    fig.tight_layout()
    return prepare_figure(fig)


def length_bins_bar(bins_df: pd.DataFrame) -> plt.Figure:
    """Bar chart with bins in logical length order (not alphabetical)."""
    fig, ax = plt.subplots(figsize=(8, 4), facecolor="white")
    ax.set_facecolor("white")
    ax.bar(bins_df["bin"].astype(str), bins_df["count"], color="steelblue")
    ax.set_xlabel("Comment length (characters)")
    ax.set_ylabel("Count")
    ax.set_title("Comment length distribution (binned)")
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    return prepare_figure(fig)


def comment_score_bins_bar(
    df: pd.DataFrame,
    title: str = "Distribution of comment scores in the cleaned Reddit dataset",
) -> plt.Figure | None:
    """Binned comment-score bar chart with horizontal tick labels."""
    if df.empty or "comment_score" not in df.columns:
        return None

    score_bins = pd.cut(
        df["comment_score"],
        bins=[-100, -1, 0, 1, 5, 20, 1000],
        labels=["<-1", "-1 to 0", "1", "2-5", "6-20", "20+"],
    )
    counts = score_bins.value_counts().sort_index()
    if counts.empty:
        return None

    labels = counts.index.astype(str).tolist()
    fig, ax = plt.subplots(figsize=(12, 6), facecolor="white", dpi=150)
    ax.set_facecolor("white")
    bars = ax.bar(labels, counts.values, color="steelblue", width=0.65, edgecolor="white", linewidth=1.2)
    ax.bar_label(bars, fmt="%d", padding=3, fontsize=10)
    ax.set_xlabel("Comment score bucket", fontsize=12)
    ax.set_ylabel("Number of comments", fontsize=12)
    ax.set_title(title, fontsize=14, pad=14)
    ax.tick_params(axis="x", labelrotation=0, labelsize=11)
    ax.tick_params(axis="y", labelsize=11)
    ax.margins(x=0.05)
    fig.tight_layout()
    return prepare_figure(fig)


def top_terms_bar(terms: pd.DataFrame, title: str = "Most frequent words", value_col: str = "count", label_col: str = "term") -> plt.Figure:
    plot_df = terms.sort_values(value_col)
    fig, ax = plt.subplots(figsize=(8, max(4, len(plot_df) * 0.25)), facecolor="white")
    ax.set_facecolor("white")
    ax.barh(plot_df[label_col].astype(str), plot_df[value_col], color="teal")
    ax.set_xlabel("Count")
    ax.set_title(title)
    fig.tight_layout()
    return prepare_figure(fig)


def monthly_activity(monthly: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(9, 4), facecolor="white")
    ax.set_facecolor("white")
    ax.plot(monthly["month"], monthly["comments"], marker="o", label="Comments", color="steelblue")
    ax.plot(monthly["month"], monthly["posts"], marker="s", label="Posts", color="darkorange")
    ax.set_xlabel("Month")
    ax.set_ylabel("Count")
    ax.set_title("Monthly activity")
    ax.legend()
    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()
    return prepare_figure(fig)


def weekday_bar(weekday_df: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 4), facecolor="white")
    ax.set_facecolor("white")
    ax.bar(weekday_df["weekday"].astype(str), weekday_df["comments"], color="slateblue")
    ax.set_xlabel("Day of week")
    ax.set_ylabel("Comments")
    ax.set_title("Comments by day of week")
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    return prepare_figure(fig)


def sentiment_bar(sentiment_df: pd.DataFrame) -> plt.Figure:
    plot_df = sentiment_df.sort_values("comments")
    colors = {"negative": "#d62728", "neutral": "#7f7f7f", "positive": "#2ca02c"}
    bar_colors = [colors.get(label, "steelblue") for label in plot_df["sentiment"]]
    fig, ax = plt.subplots(figsize=(8, 4), facecolor="white")
    ax.set_facecolor("white")
    ax.barh(plot_df["sentiment"], plot_df["comments"], color=bar_colors)
    ax.set_xlabel("Comments")
    ax.set_ylabel("Sentiment")
    ax.set_title("Predicted sentiment distribution")
    fig.tight_layout()
    return prepare_figure(fig)


def monthly_sentiment_lines(trend_df: pd.DataFrame) -> plt.Figure | None:
    if trend_df.empty:
        return None

    colors = {"negative": "#d62728", "neutral": "#7f7f7f", "positive": "#2ca02c"}
    fig, ax = plt.subplots(figsize=(9, 4), facecolor="white")
    ax.set_facecolor("white")
    for sentiment, group in trend_df.groupby("predicted_sentiment"):
        ax.plot(
            group["month"],
            group["share"],
            marker="o",
            label=sentiment,
            color=colors.get(sentiment, "steelblue"),
        )
    ax.set_xlabel("Month")
    ax.set_ylabel("Share of comments")
    ax.set_title("Monthly predicted sentiment share")
    ax.legend()
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    return prepare_figure(fig)


def sentiment_pie(sentiment_df: pd.DataFrame) -> plt.Figure:
    colors = {"negative": "#d62728", "neutral": "#7f7f7f", "positive": "#2ca02c"}
    pie_colors = [colors.get(label, "steelblue") for label in sentiment_df["sentiment"]]
    fig, ax = plt.subplots(figsize=(6, 5), facecolor="white")
    ax.set_facecolor("white")
    ax.pie(
        sentiment_df["comments"],
        labels=sentiment_df["sentiment"],
        colors=pie_colors,
        autopct="%1.1f%%",
        startangle=90,
    )
    ax.set_title("Predicted sentiment share")
    fig.tight_layout()
    return prepare_figure(fig)


def monthly_sentiment_stacked(monthly_df: pd.DataFrame) -> plt.Figure | None:
    if monthly_df.empty:
        return None

    colors = {"negative": "#d62728", "neutral": "#7f7f7f", "positive": "#2ca02c"}
    pivot = monthly_df.pivot(index="month", columns="predicted_sentiment", values="comments").fillna(0)
    for label in ["negative", "neutral", "positive"]:
        if label not in pivot.columns:
            pivot[label] = 0
    pivot = pivot[["negative", "neutral", "positive"]]

    fig, ax = plt.subplots(figsize=(9, 4), facecolor="white")
    ax.set_facecolor("white")
    bottom = None
    for label in pivot.columns:
        values = pivot[label]
        ax.bar(
            pivot.index.astype(str),
            values,
            bottom=bottom,
            label=label,
            color=colors.get(label, "steelblue"),
        )
        bottom = values if bottom is None else bottom + values
    ax.set_xlabel("Month")
    ax.set_ylabel("Comments")
    ax.set_title("Monthly predicted sentiment volume")
    ax.legend()
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    return prepare_figure(fig)


def sentiment_by_score_bar(score_df: pd.DataFrame) -> plt.Figure | None:
    if score_df.empty:
        return None

    colors = {"negative": "#d62728", "neutral": "#7f7f7f", "positive": "#2ca02c"}
    plot_df = score_df.set_index("score_bucket")
    fig, ax = plt.subplots(figsize=(9, 4), facecolor="white")
    ax.set_facecolor("white")
    x = range(len(plot_df.index))
    width = 0.25
    for idx, label in enumerate(["negative", "neutral", "positive"]):
        if label not in plot_df.columns:
            continue
        offsets = [pos + (idx - 1) * width for pos in x]
        ax.bar(offsets, plot_df[label], width=width, label=label, color=colors[label])
    ax.set_xticks(list(x))
    ax.set_xticklabels(plot_df.index.astype(str), rotation=30, ha="right")
    ax.set_xlabel("Comment score bucket")
    ax.set_ylabel("Comments")
    ax.set_title("Predicted sentiment by comment score")
    ax.legend()
    fig.tight_layout()
    return prepare_figure(fig)


def avg_score_by_sentiment_bar(score_df: pd.DataFrame) -> plt.Figure | None:
    if score_df.empty:
        return None

    colors = {"negative": "#d62728", "neutral": "#7f7f7f", "positive": "#2ca02c"}
    fig, ax = plt.subplots(figsize=(7, 4), facecolor="white")
    ax.set_facecolor("white")
    bar_colors = [colors.get(label, "steelblue") for label in score_df["predicted_sentiment"]]
    ax.bar(score_df["predicted_sentiment"], score_df["mean_score"], color=bar_colors)
    ax.set_xlabel("Predicted sentiment")
    ax.set_ylabel("Mean comment score")
    ax.set_title("Mean Reddit score by predicted sentiment")
    fig.tight_layout()
    return prepare_figure(fig)


def all_emotions_bar(emotion_df: pd.DataFrame, title: str = "All predicted emotion labels") -> plt.Figure | None:
    if emotion_df.empty:
        return None

    group_colors = {
        "positive": "#2ca02c",
        "negative": "#d62728",
        "neutral": "#7f7f7f",
        "ambiguous": "#bcbd22",
        "unmapped": "#1f77b4",
    }
    plot_df = emotion_df.sort_values("comments")
    if "sentiment_group" in plot_df.columns:
        colors = [group_colors.get(g, "steelblue") for g in plot_df["sentiment_group"]]
    else:
        colors = "teal"

    height = max(6, len(plot_df) * 0.28)
    fig, ax = plt.subplots(figsize=(9, height), facecolor="white")
    ax.set_facecolor("white")
    ax.barh(plot_df["emotion"], plot_df["comments"], color=colors)
    ax.set_xlabel("Comments containing label")
    ax.set_ylabel("Emotion")
    ax.set_title(title)
    fig.tight_layout()
    return prepare_figure(fig)


def monthly_emotion_lines(trend_df: pd.DataFrame) -> plt.Figure | None:
    if trend_df.empty:
        return None

    fig, ax = plt.subplots(figsize=(9, 4), facecolor="white")
    ax.set_facecolor("white")
    for emotion, group in trend_df.groupby("emotion"):
        ax.plot(group["month"], group["share"], marker="o", label=emotion)
    ax.set_xlabel("Month")
    ax.set_ylabel("Share of comments (within month)")
    ax.set_title("Monthly emotion share (top labels)")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    return prepare_figure(fig)


def emotion_cooccurrence_bar(pairs_df: pd.DataFrame) -> plt.Figure | None:
    if pairs_df.empty:
        return None

    plot_df = pairs_df.sort_values("comments")
    fig, ax = plt.subplots(figsize=(8, max(4, len(plot_df) * 0.3)), facecolor="white")
    ax.set_facecolor("white")
    ax.barh(plot_df["pair"], plot_df["comments"], color="mediumpurple")
    ax.set_xlabel("Comments")
    ax.set_title("Top emotion label co-occurrences")
    fig.tight_layout()
    return prepare_figure(fig)


def emotion_cooccurrence_heatmap(matrix_df: pd.DataFrame) -> plt.Figure | None:
    if matrix_df.empty:
        return None

    size = max(6, 0.45 * len(matrix_df))
    fig, ax = plt.subplots(figsize=(size + 1.5, size), facecolor="white")
    ax.set_facecolor("white")
    image = ax.imshow(matrix_df.values, cmap="YlOrRd", aspect="equal")
    ax.set_xticks(range(len(matrix_df.columns)))
    ax.set_yticks(range(len(matrix_df.index)))
    ax.set_xticklabels(matrix_df.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(matrix_df.index, fontsize=8)
    ax.set_title("Emotion co-occurrence heatmap")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Comments")
    fig.tight_layout()
    return prepare_figure(fig)


def sentiment_vs_post_score_scatter(df: pd.DataFrame, max_points: int = 4000) -> plt.Figure | None:
    if df.empty or "post_score" not in df.columns:
        return None

    colors = {"negative": "#d62728", "neutral": "#7f7f7f", "positive": "#2ca02c"}
    working = df.dropna(subset=["post_score", "predicted_sentiment"]).copy()
    if working.empty:
        return None

    if len(working) > max_points:
        working = working.sample(max_points, random_state=42)

    fig, ax = plt.subplots(figsize=(9, 4.5), facecolor="white")
    ax.set_facecolor("white")
    rng = np.random.default_rng(42)
    y_base = {"negative": 0, "neutral": 1, "positive": 2}
    for sentiment in ["negative", "neutral", "positive"]:
        subset = working[working["predicted_sentiment"] == sentiment]
        if subset.empty:
            continue
        jitter = rng.normal(0, 0.08, size=len(subset))
        ax.scatter(
            subset["post_score"],
            y_base[sentiment] + jitter,
            s=12,
            alpha=0.35,
            color=colors[sentiment],
            label=sentiment,
        )
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(["negative", "neutral", "positive"])
    ax.set_xlabel("Post score")
    ax.set_ylabel("Predicted sentiment")
    ax.set_title("Predicted sentiment vs post score")
    ax.legend(loc="upper right")
    fig.tight_layout()
    return prepare_figure(fig)


def comment_score_violin_by_sentiment(df: pd.DataFrame) -> plt.Figure | None:
    if df.empty or "comment_score" not in df.columns:
        return None

    colors = {"negative": "#d62728", "neutral": "#7f7f7f", "positive": "#2ca02c"}
    data = []
    labels = []
    plot_colors = []
    for sentiment in ["negative", "neutral", "positive"]:
        scores = df.loc[df["predicted_sentiment"] == sentiment, "comment_score"].dropna()
        if scores.empty:
            continue
        data.append(scores.to_numpy())
        labels.append(sentiment)
        plot_colors.append(colors[sentiment])

    if not data:
        return None

    fig, ax = plt.subplots(figsize=(8, 4.5), facecolor="white")
    ax.set_facecolor("white")
    parts = ax.violinplot(data, showmeans=True, showmedians=True, showextrema=True)
    for body, color in zip(parts["bodies"], plot_colors):
        body.set_facecolor(color)
        body.set_alpha(0.55)
    for key in ("cbars", "cmins", "cmaxes", "cmeans", "cmedians"):
        if key in parts:
            parts[key].set_color("#333333")
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels)
    ax.set_xlabel("Predicted sentiment")
    ax.set_ylabel("Comment score")
    ax.set_title("Comment score distribution by predicted sentiment")
    fig.tight_layout()
    return prepare_figure(fig)


def wordcloud_figure(df: pd.DataFrame, max_words: int = 100) -> plt.Figure | None:
    freqs_df = word_frequencies(df, n=max_words * 2)
    if freqs_df.empty:
        return None

    freq_map = dict(zip(freqs_df["term"], freqs_df["count"]))
    cloud = WordCloud(
        width=1000,
        height=500,
        background_color="white",
        max_words=max_words,
        colormap="viridis",
    ).generate_from_frequencies(freq_map)

    fig, ax = plt.subplots(figsize=(10, 5), facecolor="white")
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.imshow(cloud, interpolation="bilinear")
    ax.axis("off")
    ax.set_title("Word cloud (cleaned comments)", color="black")
    fig.tight_layout()
    return prepare_figure(fig)
