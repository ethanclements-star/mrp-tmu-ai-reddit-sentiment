"""Streamlit UI for Reddit EDA and (future) experiments."""

import sys
from pathlib import Path

# Always prefer the local src/ copy over an older installed package.
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) in sys.path:
    sys.path.remove(str(SRC))
sys.path.insert(0, str(SRC))

# Streamlit caches imports between reruns — reload local package after code changes.
for module_name in list(sys.modules):
    if module_name == "reddit_eda" or module_name.startswith("reddit_eda."):
        del sys.modules[module_name]

import pandas as pd
import streamlit as st

from reddit_eda.clean import CleaningOptions, apply_cleaning, cleaning_funnel
from reddit_eda.config import DEFAULT_DATASET, EXPERIMENTS_DIR, REAL_DATA_DIR
from reddit_eda.lexicon_analysis import (
    emotional_summary,
    emotional_word_counts,
    keyword_frequency_table,
)
from reddit_eda.load import load_dataset
from reddit_eda.plots import (
    all_emotions_bar,
    avg_score_by_sentiment_bar,
    comment_length_histogram,
    comment_score_bins_bar,
    comment_score_violin_by_sentiment,
    emotion_cooccurrence_bar,
    emotion_cooccurrence_heatmap,
    fig_to_png_bytes,
    length_bins_bar,
    monthly_activity,
    monthly_emotion_lines,
    monthly_sentiment_lines,
    monthly_sentiment_stacked,
    prepare_figure,
    sentiment_bar,
    sentiment_by_score_bar,
    sentiment_pie,
    sentiment_vs_post_score_scatter,
    top_terms_bar,
    weekday_bar,
    wordcloud_figure,
)
from reddit_eda.experiments import list_experiments, predictions_path
from reddit_eda.predictions import (
    attach_post_scores,
    avg_post_score_by_sentiment,
    avg_score_by_sentiment,
    emotion_cooccurrence,
    emotion_cooccurrence_matrix,
    emotion_count_per_comment,
    emotion_frequency_all,
    emotion_frequency_table,
    emotions_with_sentiment_mapping,
    filter_predictions,
    load_experiment_metadata,
    load_predictions,
    monthly_emotion_share,
    monthly_emotion_trend,
    monthly_sentiment_share,
    monthly_sentiment_trend,
    sentiment_by_score_bucket,
    sentiment_distribution,
)
from reddit_eda.summarize import dataset_summary, length_bins, monthly_volume, top_terms
from reddit_eda.text_analysis import (
    bigram_frequencies,
    comments_containing_term,
    emoji_frequencies,
    top_authors,
    weekday_volume,
    word_frequencies,
)


@st.cache_data(show_spinner="Loading dataset...")
def cached_load(path: str) -> pd.DataFrame:
    return load_dataset(path)


def _fmt_dt(value) -> str:
    if value is None or pd.isna(value):
        return "—"
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def _summary_columns(summary: dict) -> None:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Comments", f"{summary['comments']:,}")
    c2.metric("Posts", f"{summary['posts']:,}")
    c3.metric("Authors", f"{summary['authors']:,}")
    c4.metric("Median length", f"{summary['median_comment_length']:.0f} chars")

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Mean comment score", f"{summary['mean_comment_score']:.2f}")
    c6.metric("Mean post score", f"{summary['mean_post_score']:.1f}")
    c7.metric("Comments / post (median)", f"{summary['median_comments_per_post']:.1f}")
    c8.metric("AutoMod share", f"{summary['automod_share'] * 100:.1f}%")


def _render_overview(raw_df: pd.DataFrame, clean_df: pd.DataFrame) -> None:
    st.subheader("Dataset overview")
    st.caption(f"Source: `{DEFAULT_DATASET.name}` · Reddit visual-art sample · full 2025")

    raw_summary = dataset_summary(raw_df)
    cleaned_summary = dataset_summary(clean_df)

    left, right = st.columns(2)
    with left:
        st.markdown("**Raw dataset**")
        _summary_columns(raw_summary)
        st.write(
            f"Post dates: {_fmt_dt(raw_summary['post_start'])} → {_fmt_dt(raw_summary['post_end'])}  \n"
            f"Comment dates: {_fmt_dt(raw_summary['comment_start'])} → {_fmt_dt(raw_summary['comment_end'])}"
        )

    with right:
        st.markdown("**After cleaning**")
        _summary_columns(cleaned_summary)
        removed = raw_summary["comments"] - cleaned_summary["comments"]
        pct = (removed / raw_summary["comments"] * 100) if raw_summary["comments"] else 0
        st.write(f"Removed **{removed:,}** rows ({pct:.1f}%) with current filters.")


def _render_cleaning_tab(raw_df: pd.DataFrame, options: CleaningOptions) -> pd.DataFrame:
    st.subheader("Cleaning pipeline")
    funnel = cleaning_funnel(raw_df, options)
    clean_df = apply_cleaning(raw_df, options)

    chart_df = funnel.set_index("step")[["rows"]]
    st.bar_chart(chart_df)
    st.dataframe(funnel, use_container_width=True, hide_index=True)

    st.caption(
        "Rows removed at each step. AutoModerator messages are a large share of this subreddit."
    )
    return clean_df


def _render_text_tab(clean_df: pd.DataFrame) -> None:
    st.subheader("Comment text")
    if clean_df.empty:
        st.warning("No rows left after cleaning. Relax the filters in the sidebar.")
        return

    left, right = st.columns(2)

    with left:
        st.markdown("**Length distribution (characters)**")
        bins_df = length_bins(clean_df)
        if not bins_df.empty:
            _show_figure(length_bins_bar(bins_df))

    with right:
        st.markdown("**Comments per month**")
        monthly = monthly_volume(clean_df)
        if not monthly.empty:
            st.line_chart(monthly.set_index("month")[["comments"]])

    st.markdown("**Most frequent terms** (basic tokenization, stopwords removed)")
    terms = top_terms(clean_df, n=20)
    if not terms.empty:
        st.bar_chart(terms.set_index("term"))


def _render_engagement_tab(clean_df: pd.DataFrame) -> None:
    st.subheader("Engagement")
    if clean_df.empty:
        st.warning("No rows left after cleaning.")
        return

    st.markdown("**Comment score distribution**")
    fig = comment_score_bins_bar(clean_df)
    if fig is not None:
        _show_figure(fig)
        _download_figure_button(fig, "comment_score_distribution.png")
    else:
        st.info("No comment scores available.")

    left, right = st.columns(2)

    with left:
        st.markdown("**Post score vs comments sampled**")
        post_stats = (
            clean_df.groupby("post_id", as_index=False)
            .agg(post_score=("post_score", "first"), comments=("comment_id", "count"))
        )
        st.scatter_chart(
            post_stats,
            x="post_score",
            y="comments",
        )

    with right:
        st.markdown("**Top posts by comment count (in cleaned data)**")
        top_posts = (
            clean_df.groupby(["post_id", "post_title"], as_index=False)
            .agg(comments=("comment_id", "count"), post_score=("post_score", "first"))
            .sort_values("comments", ascending=False)
            .head(10)
        )
        st.dataframe(top_posts, use_container_width=True, hide_index=True)


def _render_preview_tab(clean_df: pd.DataFrame) -> None:
    st.subheader("Data preview")
    cols = [
        "post_title",
        "comment_body",
        "comment_score",
        "comment_author",
        "comment_length",
        "post_score",
        "comment_created_at",
    ]
    available = [c for c in cols if c in clean_df.columns]
    st.dataframe(
        clean_df[available].head(200),
        use_container_width=True,
        hide_index=True,
    )


def _show_figure(fig) -> None:
    """Display matplotlib figure with a white background."""
    if fig is None:
        return
    prepare_figure(fig)
    st.pyplot(fig, use_container_width=True)


def _download_csv_button(df: pd.DataFrame, filename: str, label: str) -> None:
    if df.empty:
        return
    st.download_button(
        label=label,
        data=df.to_csv(index=False).encode("utf-8"),
        file_name=filename,
        mime="text/csv",
    )


def _download_figure_button(fig, filename: str) -> None:
    if fig is None:
        return
    st.download_button(
        label="Download chart (PNG)",
        data=fig_to_png_bytes(fig),
        file_name=filename,
        mime="image/png",
    )


def _render_explore_tab(clean_df: pd.DataFrame) -> None:
    st.subheader("Interactive exploration")
    st.caption("Charts and tables use the **cleaned** dataset with your current sidebar filters.")

    if clean_df.empty:
        st.warning("No rows left after cleaning. Relax the filters in the sidebar.")
        return

    chart_options = [
        "Word cloud",
        "Top words (table + chart)",
        "Top word pairs (bigrams)",
        "Topic keyword frequencies",
        "Emotional word analysis",
        "Emoji frequency",
        "Comment length histogram",
        "Monthly activity",
        "Day of week",
        "Top authors",
        "Search comments by word",
    ]

    left, right = st.columns([1, 2])
    with left:
        chart = st.selectbox("Visualization", chart_options)
        top_n = st.slider("How many items to show", 10, 100, 30, step=5)
        max_words = st.slider("Word cloud size", 50, 200, 100, step=10)

    with right:
        if chart == "Word cloud":
            fig = wordcloud_figure(clean_df, max_words=max_words)
            if fig:
                _show_figure(fig)
                _download_figure_button(fig, "wordcloud.png")
            else:
                st.info("Not enough text to build a word cloud.")

        elif chart == "Top words (table + chart)":
            terms = word_frequencies(clean_df, n=top_n)
            terms["share_pct"] = (terms["share"] * 100).round(2)
            st.dataframe(
                terms[["term", "count", "share_pct"]].rename(columns={"share_pct": "share_%"}),
                use_container_width=True,
                hide_index=True,
            )
            fig = top_terms_bar(terms.head(min(top_n, 25)))
            _show_figure(fig)
            _download_csv_button(terms, "top_words.csv", "Download word list (CSV)")
            _download_figure_button(fig, "top_words.png")

        elif chart == "Top word pairs (bigrams)":
            bigrams = bigram_frequencies(clean_df, n=top_n)
            st.dataframe(bigrams, use_container_width=True, hide_index=True)
            if not bigrams.empty:
                plot_df = bigrams.head(min(top_n, 20)).rename(columns={"bigram": "term"})
                fig = top_terms_bar(plot_df, title="Most frequent word pairs")
                _show_figure(fig)
                _download_csv_button(bigrams, "top_bigrams.csv", "Download bigrams (CSV)")
                _download_figure_button(fig, "top_bigrams.png")

        elif chart == "Topic keyword frequencies":
            keywords = keyword_frequency_table(clean_df)
            st.dataframe(keywords, use_container_width=True, hide_index=True)
            if not keywords.empty:
                plot_df = keywords.head(min(top_n, 20)).rename(columns={"keyword": "term"})
                fig = top_terms_bar(
                    plot_df,
                    title="Comments containing tracked topic keywords",
                    value_col="comments_with_keyword",
                    label_col="term",
                )
                _show_figure(fig)
                _download_csv_button(keywords, "topic_keywords.csv", "Download keyword table (CSV)")
                _download_figure_button(fig, "topic_keywords.png")
            st.caption(
                "Counts how many comments mention each keyword at least once, "
                "plus total mentions across the corpus."
            )

        elif chart == "Emotional word analysis":
            summary = emotional_summary(clean_df)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Positive word hits", f"{summary['positive_mentions']:,}")
            c2.metric("Negative word hits", f"{summary['negative_mentions']:,}")
            c3.metric("Comments w/ positive words", f"{summary['pct_positive_comments']}%")
            c4.metric("Comments w/ negative words", f"{summary['pct_negative_comments']}%")

            emotional = emotional_word_counts(clean_df)
            st.markdown("**Emotionally charged word counts** (basic lexicon)")
            if emotional.empty:
                st.info("No emotional lexicon words found.")
            else:
                st.dataframe(emotional, use_container_width=True, hide_index=True)
                plot_df = emotional.head(min(top_n, 25)).rename(columns={"word": "term"})
                fig = top_terms_bar(plot_df, title="Top emotional words", label_col="term")
                _show_figure(fig)
                _download_csv_button(emotional, "emotional_words.csv", "Download emotional words (CSV)")
                _download_figure_button(fig, "emotional_words.png")
            st.caption(
                "Simple dictionary-based counts — not a trained sentiment model. "
                "Useful for exploratory signals only."
            )

        elif chart == "Emoji frequency":
            emojis = emoji_frequencies(clean_df, n=top_n)
            st.dataframe(emojis, use_container_width=True, hide_index=True)
            if not emojis.empty:
                st.bar_chart(emojis.set_index("emoji"))
                _download_csv_button(emojis, "emoji_frequency.csv", "Download emoji counts (CSV)")
            else:
                st.info("No emojis found in cleaned comments.")

        elif chart == "Comment length histogram":
            fig = comment_length_histogram(clean_df)
            _show_figure(fig)
            _download_figure_button(fig, "comment_length_histogram.png")

        elif chart == "Monthly activity":
            monthly = monthly_volume(clean_df)
            fig = monthly_activity(monthly)
            _show_figure(fig)
            st.dataframe(monthly, use_container_width=True, hide_index=True)
            _download_csv_button(monthly, "monthly_activity.csv", "Download monthly table (CSV)")
            _download_figure_button(fig, "monthly_activity.png")

        elif chart == "Day of week":
            weekday = weekday_volume(clean_df)
            fig = weekday_bar(weekday)
            _show_figure(fig)
            st.dataframe(weekday, use_container_width=True, hide_index=True)
            _download_csv_button(weekday, "weekday_volume.csv", "Download weekday table (CSV)")
            _download_figure_button(fig, "weekday_volume.png")

        elif chart == "Top authors":
            authors = top_authors(clean_df, n=top_n)
            st.dataframe(authors, use_container_width=True, hide_index=True)
            st.bar_chart(authors.set_index("comment_author")[["comments"]])
            _download_csv_button(authors, "top_authors.csv", "Download author table (CSV)")

        elif chart == "Search comments by word":
            term = st.text_input("Search for word", value="art").strip().lower()
            if term:
                examples, total = comments_containing_term(clean_df, term, limit=15)
                st.write(f"**{total:,}** matching comments (showing up to 15)")
                st.dataframe(examples, use_container_width=True, hide_index=True)


def _render_experiments_tab() -> None:
    st.subheader("Experiments")
    st.caption(
        "Explore precomputed model runs from `output/experiments/`. "
        "Charts and tables can be downloaded for your report."
    )

    runs = list_experiments()
    if not runs:
        st.warning("No experiment runs found.")
        st.code(
            "cd MRP_TEST\n"
            "source .venv/bin/activate\n"
            "python -m reddit_eda.score\n\n"
            "# Quick test on 100 comments:\n"
            "python -m reddit_eda.score --max-samples 100",
            language="bash",
        )
        st.markdown(
            f"Each run creates a folder under `{EXPERIMENTS_DIR}/` "
            f"(e.g. `reddit_roberta_YYYYMMDD_HHMMSS/predictions.csv`)."
        )
        return

    run_labels = {path.name: path for path in runs}
    selected_name = st.selectbox("Experiment run", list(run_labels.keys()))
    experiment_dir = run_labels[selected_name]
    predictions = load_predictions(predictions_path(experiment_dir))

    # Join post scores from the source Reddit dataset when missing from the experiment CSV.
    if "post_score" not in predictions.columns and DEFAULT_DATASET.exists():
        predictions = attach_post_scores(predictions, cached_load(str(DEFAULT_DATASET)))

    meta = load_experiment_metadata(experiment_dir)
    if meta:
        st.caption(
            f"Model: **{meta.get('model_label', '—')}** · "
            f"Scored: **{meta.get('comments_scored', len(predictions)):,}** comments · "
            f"Runtime: **{meta.get('runtime_sec', '—')}s** · "
            f"Folder: `{experiment_dir.name}/`"
        )

    sentiment_df = sentiment_distribution(predictions)
    all_emotions = emotion_frequency_all(predictions)
    mapped_emotions = emotions_with_sentiment_mapping(predictions)
    top_emotions = emotion_frequency_table(predictions, n=30)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Comments scored", f"{len(predictions):,}")
    c2.metric("Unique emotions predicted", f"{(all_emotions['comments'] > 0).sum()}")
    c3.metric("Top emotion", top_emotions.iloc[0]["emotion"] if not top_emotions.empty else "—")
    c4.metric(
        "Top emotion share",
        f"{top_emotions.iloc[0]['share'] * 100:.1f}%" if not top_emotions.empty else "—",
    )

    analysis_type = st.radio(
        "Analysis level",
        ["Emotions (28 labels)", "Sentiment (3 classes)"],
        horizontal=True,
    )

    if analysis_type == "Emotions (28 labels)":
        chart_options = [
            "All 28 emotion labels",
            "Emotions by sentiment mapping",
            "Top predicted emotions",
            "Monthly trend (top emotions)",
            "Emotion co-occurrences",
            "Emotion co-occurrence heatmap",
            "Labels per comment",
            "Browse by emotion",
        ]
    else:
        chart_options = [
            "Sentiment distribution (bar)",
            "Sentiment share (pie)",
            "Monthly sentiment share",
            "Monthly sentiment volume",
            "Sentiment by comment score",
            "Mean score by sentiment",
            "Sentiment vs post score",
            "Comment score violin by sentiment",
            "Browse by sentiment",
        ]

    left, right = st.columns([1, 2])
    with left:
        chart = st.selectbox("Visualization", chart_options)
        top_n = st.slider("How many items to show", 10, 28, 20, step=2)

    with right:
        if chart == "All 28 emotion labels":
            table = mapped_emotions.copy()
            table["share_pct"] = (table["share"] * 100).round(2)
            st.dataframe(
                table.rename(columns={"share_pct": "share_%"}),
                use_container_width=True,
                hide_index=True,
            )
            fig = all_emotions_bar(mapped_emotions, title="All 28 GoEmotions labels (colored by sentiment mapping)")
            _show_figure(fig)
            st.caption("Green = positive mapping · Red = negative · Gray = neutral/ambiguous")
            _download_csv_button(table, "all_emotions.csv", "Download table (CSV)")
            _download_figure_button(fig, "all_emotions.png")

        elif chart == "Emotions by sentiment mapping":
            table = mapped_emotions.copy()
            table["share_pct"] = (table["share"] * 100).round(2)
            for group in ["positive", "negative", "neutral"]:
                group_df = table[table["sentiment_group"] == group]
                if not group_df.empty:
                    st.markdown(f"**{group.title()}** ({group_df['comments'].sum():,} label hits)")
                    st.dataframe(
                        group_df[["emotion", "comments", "share_pct"]].rename(columns={"share_pct": "share_%"}),
                        use_container_width=True,
                        hide_index=True,
                    )
            fig = all_emotions_bar(mapped_emotions, title="Emotions colored by mapped sentiment group")
            _show_figure(fig)
            _download_csv_button(table, "emotions_by_mapping.csv", "Download table (CSV)")
            _download_figure_button(fig, "emotions_by_mapping.png")

        elif chart == "Top predicted emotions":
            emotions = top_emotions.head(top_n).copy()
            emotions["share_pct"] = (emotions["share"] * 100).round(2)
            st.dataframe(
                emotions.rename(columns={"share_pct": "share_%"}),
                use_container_width=True,
                hide_index=True,
            )
            plot_df = emotions.rename(columns={"emotion": "term"})
            fig = top_terms_bar(
                plot_df,
                title="Comments containing each emotion label",
                value_col="comments",
                label_col="term",
            )
            _show_figure(fig)
            _download_csv_button(emotions, "emotion_frequencies.csv", "Download table (CSV)")
            _download_figure_button(fig, "top_emotions.png")

        elif chart == "Monthly trend (top emotions)":
            selected_emotions = top_emotions.head(top_n)["emotion"].tolist()
            monthly_table = monthly_emotion_share(predictions, emotions=selected_emotions)
            fig = monthly_emotion_lines(monthly_table)
            if fig:
                _show_figure(fig)
                _download_figure_button(fig, "monthly_emotions.png")
            st.dataframe(monthly_table, use_container_width=True, hide_index=True)
            _download_csv_button(monthly_table, "monthly_emotions.csv", "Download table (CSV)")

        elif chart == "Emotion co-occurrences":
            pairs = emotion_cooccurrence(predictions, n=top_n)
            st.dataframe(pairs, use_container_width=True, hide_index=True)
            fig = emotion_cooccurrence_bar(pairs)
            if fig:
                _show_figure(fig)
                _download_figure_button(fig, "emotion_cooccurrence.png")
            _download_csv_button(pairs, "emotion_cooccurrence.csv", "Download table (CSV)")
            st.caption("Pairs of emotion labels assigned to the same comment.")

        elif chart == "Emotion co-occurrence heatmap":
            matrix = emotion_cooccurrence_matrix(predictions, top_n=min(top_n, 15))
            if matrix.empty:
                st.info("Not enough co-occurring emotion labels to build a heatmap.")
            else:
                st.dataframe(matrix, use_container_width=True)
                fig = emotion_cooccurrence_heatmap(matrix)
                if fig:
                    _show_figure(fig)
                    _download_figure_button(fig, "emotion_cooccurrence_heatmap.png")
                _download_csv_button(matrix.reset_index(names="emotion"), "emotion_cooccurrence_matrix.csv", "Download matrix (CSV)")
                st.caption(
                    "Diagonal = comments with that label. Off-diagonal = comments where both labels co-occur. "
                    "Uses the top-N most frequent emotions."
                )

        elif chart == "Labels per comment":
            label_counts = emotion_count_per_comment(predictions)
            st.dataframe(label_counts, use_container_width=True, hide_index=True)
            st.bar_chart(label_counts.set_index("labels_per_comment"))
            _download_csv_button(label_counts, "labels_per_comment.csv", "Download table (CSV)")
            st.caption("How many emotion labels RoBERTa assigned per comment (multi-label).")

        elif chart == "Browse by emotion":
            filter_left, filter_right = st.columns(2)
            with filter_left:
                emotion_options = all_emotions[all_emotions["comments"] > 0]["emotion"].tolist()
                emotion_filter = st.selectbox("Emotion label", ["All"] + emotion_options)
            with filter_right:
                sentiment_filter = st.selectbox(
                    "Also filter by mapped sentiment", ["All", "positive", "neutral", "negative"]
                )

            filtered = filter_predictions(predictions, sentiment_filter, emotion_filter)
            preview_cols = [
                "comment_body",
                "predicted_emotions",
                "predicted_sentiment",
                "comment_score",
                "comment_author",
                "comment_created_at",
            ]
            available = [c for c in preview_cols if c in filtered.columns]
            st.write(f"Showing **{min(200, len(filtered)):,}** of **{len(filtered):,}** matching comments")
            st.dataframe(filtered[available].head(200), use_container_width=True, hide_index=True)
            _download_csv_button(filtered, "filtered_predictions.csv", "Download filtered predictions (CSV)")

        elif chart == "Sentiment distribution (bar)":
            table = sentiment_df.copy()
            table["share_pct"] = (table["share"] * 100).round(1)
            st.dataframe(
                table.rename(columns={"share_pct": "share_%"}),
                use_container_width=True,
                hide_index=True,
            )
            fig = sentiment_bar(sentiment_df)
            _show_figure(fig)
            _download_csv_button(table, "sentiment_distribution.csv", "Download table (CSV)")
            _download_figure_button(fig, "sentiment_distribution.png")

        elif chart == "Sentiment share (pie)":
            table = sentiment_df.copy()
            table["share_pct"] = (table["share"] * 100).round(1)
            fig = sentiment_pie(sentiment_df)
            _show_figure(fig)
            _download_csv_button(table, "sentiment_share.csv", "Download table (CSV)")
            _download_figure_button(fig, "sentiment_share.png")

        elif chart == "Monthly sentiment share":
            monthly_share = monthly_sentiment_share(predictions)
            fig = monthly_sentiment_lines(monthly_share)
            if fig:
                _show_figure(fig)
                _download_figure_button(fig, "monthly_sentiment_share.png")
            monthly_table = monthly_share.copy()
            monthly_table["share_pct"] = (monthly_table["share"] * 100).round(1)
            st.dataframe(
                monthly_table.rename(columns={"share_pct": "share_%"}),
                use_container_width=True,
                hide_index=True,
            )
            _download_csv_button(monthly_table, "monthly_sentiment_share.csv", "Download table (CSV)")

        elif chart == "Monthly sentiment volume":
            monthly_table = monthly_sentiment_trend(predictions)
            fig = monthly_sentiment_stacked(monthly_table)
            if fig:
                _show_figure(fig)
                _download_figure_button(fig, "monthly_sentiment_volume.png")
            st.dataframe(monthly_table, use_container_width=True, hide_index=True)
            _download_csv_button(monthly_table, "monthly_sentiment_volume.csv", "Download table (CSV)")

        elif chart == "Sentiment by comment score":
            score_table = sentiment_by_score_bucket(predictions)
            if score_table.empty:
                st.info("Comment scores not available.")
            else:
                fig = sentiment_by_score_bar(score_table)
                if fig:
                    _show_figure(fig)
                    _download_figure_button(fig, "sentiment_by_score.png")
                st.dataframe(score_table, use_container_width=True, hide_index=True)
                _download_csv_button(score_table, "sentiment_by_score.csv", "Download table (CSV)")

        elif chart == "Mean score by sentiment":
            score_summary = avg_score_by_sentiment(predictions)
            if score_summary.empty:
                st.info("Comment scores not available.")
            else:
                score_summary = score_summary.copy()
                score_summary["mean_score"] = score_summary["mean_score"].round(2)
                st.dataframe(score_summary, use_container_width=True, hide_index=True)
                fig = avg_score_by_sentiment_bar(score_summary)
                if fig:
                    _show_figure(fig)
                    _download_figure_button(fig, "mean_score_by_sentiment.png")
                _download_csv_button(score_summary, "mean_score_by_sentiment.csv", "Download table (CSV)")

        elif chart == "Sentiment vs post score":
            if "post_score" not in predictions.columns or predictions["post_score"].isna().all():
                st.info("Post scores are not available for this experiment.")
            else:
                summary = avg_post_score_by_sentiment(predictions)
                if not summary.empty:
                    summary = summary.copy()
                    summary["mean_post_score"] = summary["mean_post_score"].round(1)
                    summary["median_post_score"] = summary["median_post_score"].round(1)
                    st.dataframe(summary, use_container_width=True, hide_index=True)
                    _download_csv_button(summary, "sentiment_vs_post_score.csv", "Download table (CSV)")
                fig = sentiment_vs_post_score_scatter(predictions)
                if fig:
                    _show_figure(fig)
                    _download_figure_button(fig, "sentiment_vs_post_score.png")
                st.caption(
                    "Each point is a comment plotted by its parent post's score and predicted sentiment "
                    "(with light jitter). Sampled for display if the dataset is large."
                )

        elif chart == "Comment score violin by sentiment":
            if "comment_score" not in predictions.columns:
                st.info("Comment scores not available.")
            else:
                summary = avg_score_by_sentiment(predictions)
                if not summary.empty:
                    summary = summary.copy()
                    summary["mean_score"] = summary["mean_score"].round(2)
                    st.dataframe(summary, use_container_width=True, hide_index=True)
                fig = comment_score_violin_by_sentiment(predictions)
                if fig:
                    _show_figure(fig)
                    _download_figure_button(fig, "comment_score_violin_by_sentiment.png")
                st.caption(
                    "Violin plot shows the full comment-score distribution for each predicted sentiment "
                    "(mean and median marked)."
                )

        elif chart == "Browse by sentiment":
            filter_left, filter_right = st.columns(2)
            with filter_left:
                sentiment_filter = st.selectbox(
                    "Filter by sentiment", ["All", "positive", "neutral", "negative"]
                )
            with filter_right:
                emotion_options = (
                    ["All"] + top_emotions["emotion"].tolist() if not top_emotions.empty else ["All"]
                )
                emotion_filter = st.selectbox("Filter by emotion", emotion_options)

            filtered = filter_predictions(predictions, sentiment_filter, emotion_filter)
            preview_cols = [
                "comment_body",
                "predicted_emotions",
                "predicted_sentiment",
                "comment_score",
                "comment_author",
                "comment_created_at",
            ]
            available = [c for c in preview_cols if c in filtered.columns]
            st.write(f"Showing **{min(200, len(filtered)):,}** of **{len(filtered):,}** matching comments")
            st.dataframe(filtered[available].head(200), use_container_width=True, hide_index=True)
            _download_csv_button(filtered, "filtered_predictions.csv", "Download filtered predictions (CSV)")
            _download_csv_button(predictions, "all_predictions.csv", "Download full predictions (CSV)")


def main() -> None:
    st.set_page_config(page_title="MRP Reddit EDA", layout="wide")
    st.title("Reddit EDA — visual-art sample 2025")
    st.markdown("Explore and clean your Reddit comment dataset before running emotion models.")

    with st.sidebar:
        st.header("Cleaning options")
        remove_automod = st.checkbox("Remove AutoModerator", value=True)
        remove_deleted = st.checkbox("Remove [deleted] / [removed]", value=True)
        strip_urls_from_text = st.checkbox("Strip URLs from comment text", value=True)
        remove_empty = st.checkbox("Remove empty comments", value=True)
        dedupe = st.checkbox("Deduplicate by comment_id", value=True)
        min_length = st.slider("Minimum comment length (characters)", 0, 200, 0, step=5)

        st.divider()
        st.caption(f"Data folder: `{REAL_DATA_DIR}`")

        options = CleaningOptions(
            remove_automoderator=remove_automod,
            remove_deleted=remove_deleted,
            strip_urls=strip_urls_from_text,
            remove_empty=remove_empty,
            min_comment_length=min_length,
            dedupe_comments=dedupe,
        )

    if not DEFAULT_DATASET.exists():
        st.error(f"Dataset not found: `{DEFAULT_DATASET}`")
        st.stop()

    raw_df = cached_load(str(DEFAULT_DATASET))
    clean_df = apply_cleaning(raw_df, options)

    overview_tab, cleaning_tab, text_tab, explore_tab, engagement_tab, preview_tab, experiments_tab = st.tabs(
        ["Overview", "Cleaning", "Text", "Explore", "Engagement", "Preview", "Experiments"]
    )

    with overview_tab:
        _render_overview(raw_df, clean_df)

    with cleaning_tab:
        clean_df = _render_cleaning_tab(raw_df, options)

    with text_tab:
        _render_text_tab(clean_df)

    with explore_tab:
        _render_explore_tab(clean_df)

    with engagement_tab:
        _render_engagement_tab(clean_df)

    with preview_tab:
        _render_preview_tab(clean_df)

    with experiments_tab:
        _render_experiments_tab()


if __name__ == "__main__":
    main()
