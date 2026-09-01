from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REAL_DATA_DIR = PROJECT_ROOT / "data" / "real_data"
DEFAULT_DATASET = REAL_DATA_DIR / "reddit_art_2025.csv"
OUTPUT_DIR = PROJECT_ROOT / "output" / "eda"
EXPERIMENTS_DIR = PROJECT_ROOT / "output" / "experiments"

BOT_AUTHORS = {"AutoModerator", "automoderator"}
DELETED_PATTERNS = ("[deleted]", "[removed]")
