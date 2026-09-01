# GoEmotions Benchmark & Reddit EDA

Python tools for benchmarking emotion classifiers on GoEmotions and exploring a Reddit visual-art comment dataset in the browser.

## Quick start

```bash
cd MRP_TEST
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e .

# Put the dataset at data/real_data/reddit_art_2025.csv (see Data below), then:
streamlit run app/main.py
```

Opens the EDA dashboard at **http://localhost:8501**.

## Data

The Reddit CSV is **not committed to git** (`data/` is gitignored). After cloning, either:

1. **Place the file manually** at `data/real_data/reddit_art_2025.csv`, or
2. **Regenerate it** with the collection script (slow; requires network):

```bash
pip install requests
python scripts/get_data_reddit_art.py
```

The collector lives at **`scripts/get_data_reddit_art.py`**. It uses the [Arctic Shift API](https://arctic-shift.photon-reddit.com/) to sample ~20,000 visual posts from 2025 (evenly by week) and up to 10 comments per post. See [Reddit data collection](#reddit-data-collection-reproducibility) for full details.

## Components

| Component | Command | Purpose |
|-----------|---------|---------|
| **EDA UI** | `streamlit run app/main.py` | Explore & clean `data/real_data/` |
| **Score Reddit** | `python -m reddit_eda.score` | Run RoBERTa on cleaned comments, save CSV |
| **Benchmark** | `python -m goemotions_benchmark.run` | Compare models on GoEmotions |

---

## EDA dashboard (browser UI)

Interactive Streamlit app for `data/real_data/reddit_art_2025.csv`:

- Dataset summary (raw vs cleaned)
- Toggle cleaning: AutoModerator, deleted/removed, **URL stripping**, empty, min length, dedupe
- Charts: cleaning funnel, comment length, monthly volume, top terms, engagement
- **Explore** tab: word cloud, top words/bigrams tables, emoji frequency, histograms, downloads for report
- Data preview table
- **Experiments** tab: load saved model predictions (sentiment + emotion charts)

```bash
cd MRP_TEST
source .venv/bin/activate
pip install -e .
streamlit run app/main.py
```

Use the sidebar to adjust cleaning — all charts update live.

### Score real Reddit data (offline)

Run the GoEmotions model on **default-cleaned** comments. Each run creates an experiment folder:

```
output/experiments/reddit_roberta_YYYYMMDD_HHMMSS/
├── predictions.csv    # per-comment emotions + mapped sentiment
├── metadata.json      # model, runtime, cleaning settings
└── figures/           # optional chart exports
```

```bash
# Full cleaned dataset (~22k comments; may take 15–30+ min on CPU)
python -m reddit_eda.score

# Quick test
python -m reddit_eda.score --max-samples 100

# Named experiment folder
python -m reddit_eda.score --experiment reddit_roberta_full

# Alternative model
python -m reddit_eda.score --model bert_monologg
```

Then open the app → **Experiments** tab to explore charts and download tables.

---

## Reddit data collection (reproducibility)

The primary Reddit dataset was built with **`scripts/get_data_reddit_art.py`** using the [Arctic Shift API](https://arctic-shift.photon-reddit.com/). The script:

- Samples **~20,000 visual posts** from a visual-art community across **all of 2025**, spread evenly by week
- Keeps only image/video posts (`post_hint`: image, rich:video, hosted:video)
- Fetches up to **10 top comments** per post
- Writes **`data/real_data/reddit_art_2025.csv`**

To re-run (requires network access; takes a long time):

```bash
cd MRP_TEST
source .venv/bin/activate
pip install requests   # not required for EDA/benchmark, only for collection
python scripts/get_data_reddit_art.py
```

A comparison sample from a discussion-focused community (same year, no visual-only filter) is built with:

```bash
python scripts/get_data_reddit_wars.py
```

That writes `data/real_data/reddit_wars_2025.csv`.

---

## GoEmotions benchmark

### What it does

1. Loads the official GoEmotions **train** and **test** splits.
2. Runs each model on the **test** set (classical model trains on train first).
3. Evaluates **28 emotion labels** (multi-label).
4. Maps predictions to **3 sentiment classes** (negative / neutral / positive) using Google's official emotion-to-sentiment mapping.
5. Saves CSV/JSON results and matplotlib figures to `output/`.

### Models

| Key | Model | Notes |
|-----|-------|-------|
| `tfidf_logreg` | TF-IDF + Logistic Regression | Classical baseline; trains on train split |
| `bert_monologg` | `monologg/bert-base-cased-goemotions-original` | BERT fine-tuned on GoEmotions |
| `roberta_samlowe` | `SamLowe/roberta-base-go_emotions` | RoBERTa fine-tuned on GoEmotions |
| `ollama_llama` | Ollama (local LLM) | Zero-shot via prompt; requires [Ollama](https://ollama.com) |

### Requirements

- Python 3.11+
- ~2 GB disk for model downloads (first run only)
- **For local LLM:** [Ollama](https://ollama.com) installed and running, plus a pulled model:
  ```bash
  ollama pull llama3.2:3b
  ```

### Run

```bash
# All models on full test set
python -m goemotions_benchmark.run

# Quick smoke test (100 test comments)
python -m goemotions_benchmark.run --max-test-samples 100

# Specific models only
python -m goemotions_benchmark.run --models tfidf_logreg,roberta_samlowe

# Local LLM via Ollama (start with a small sample — one API call per comment)
python -m goemotions_benchmark.run --models ollama_llama --max-test-samples 200

# Use a different Ollama model
python -m goemotions_benchmark.run --models ollama_llama --ollama-model mistral --max-test-samples 100
```

First transformer run downloads models from Hugging Face and may take several minutes.

LLM responses are cached under `data/llm_cache/` so re-runs on the same comments are free.

### Output

Each run creates a timestamped folder under `output/`:

```
output/20250615_143022/
├── benchmark_summary.csv
├── figures/
│   ├── benchmark_macro_f1_comparison.png
│   ├── tfidf_logreg_emotion_f1_by_class.png
│   ├── tfidf_logreg_sentiment_f1_by_class.png
│   └── ...
├── tfidf_logreg/
│   ├── emotions_per_class.csv
│   ├── emotions_summary.json
│   ├── sentiment_per_class.csv
│   └── sentiment_summary.json
└── roberta_samlowe/
    └── ...
```

### Metrics

**28 emotions (multi-label)**

| Metric | Meaning |
|--------|---------|
| Macro F1 | Average F1 across all 28 emotions |
| Micro F1 | F1 pooled over all label decisions |
| Subset accuracy | Exact match of full predicted label set (strict — one wrong label fails) |
| At least one match | Share of comments where predicted and true labels overlap |
| Mean Jaccard | Average \|true ∩ pred\| / \|true ∪ pred\| across comments |
| Hamming loss | Fraction of wrong label decisions |
| Per-class P/R/F1 | One row per emotion in `emotions_per_class.csv` |

**3 sentiment (single-label, derived)**

All models use the same mapping: predicted emotions → sentiment group → one of `negative`, `neutral`, `positive`. Ambiguous emotions (surprise, curiosity, etc.) map to `neutral`. If multiple groups apply, priority is: negative > positive > neutral.

| Metric | Meaning |
|--------|---------|
| Accuracy | Exact sentiment match |
| Macro / Micro F1 | Standard single-label F1 |
| Per-class P/R/F1 | One row per sentiment in `sentiment_per_class.csv` |

### Choosing a model

Check `benchmark_summary.csv`:

- **Highest `emotion_macro_f1`** → best fine-grained emotion detection (rare emotions count equally).
- **Highest `emotion_micro_f1`** → best overall label matching (dominated by frequent emotions).
- **Highest `sentiment_macro_f1`** → best coarse sentiment after mapping.
- **`runtime_sec`** → practical cost on your hardware.

On the full GoEmotions test set, **`bert_monologg`** had the highest emotion macro-F1 (~0.50); **`roberta_samlowe`** was close (~0.45) and roughly **3× faster**. Use BERT for best accuracy; RoBERTa if speed matters more.

## Project layout

```
MRP_TEST/
├── app/
│   └── main.py              # Streamlit EDA dashboard
├── scripts/
│   ├── get_data_reddit_art.py   # Arctic Shift collector for reddit_art_2025.csv
│   └── get_data_reddit_wars.py  # Arctic Shift collector for reddit_wars_2025.csv
├── data/
│   ├── real_data/           # Reddit CSV (local only; not in git)
│   └── ...                  # HF / LLM caches
├── output/
│   ├── experiments/         # Reddit model runs (predictions.csv per folder)
│   └── ...                  # GoEmotions benchmark runs
├── src/
│   ├── goemotions_benchmark/
│   └── reddit_eda/          # EDA logic (load, clean, score, predictions)
├── pyproject.toml
└── README.md
```
