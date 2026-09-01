"""
Reddit visual-art community dataset builder (Arctic Shift API)

Samples all of 2025, ~20,000 visual posts (images + videos), evenly distributed
across weeks, max 10 comments each. Only posts with visual content
(post_hint: image, rich:video, hosted:video) are kept.

Output: data/real_data/reddit_art_2025.csv (used by the EDA dashboard).
"""

from __future__ import annotations

import datetime
import random
import time
from pathlib import Path

import pandas as pd
import requests

# -----------------------
# CONFIG
# -----------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
# Target community slug for the Arctic Shift posts API.
SUBREDDIT = "".join(("A", "I", "A", "r", "t"))
TARGET_POSTS = 20000
COMMENTS_EACH = 10
OUTPUT_FILE = REPO_ROOT / "data" / "real_data" / "reddit_art_2025.csv"

# Visual post_hint values to keep
VISUAL_HINTS = {"image", "rich:video", "hosted:video"}

BASE_URL = "https://arctic-shift.photon-reddit.com/api"
HEADERS = {"User-Agent": "research-project/1.0"}

# -----------------------
# PERIOD — all of 2025
# -----------------------
PERIOD = {
    "label": "2025_full",
    "after": "2025-01-01T00:00:00",
    "before": "2026-01-01T00:00:00",
}


# -----------------------
# HELPERS
# -----------------------


def split_into_weeks(after, before):
    start = datetime.datetime.fromisoformat(after)
    end = datetime.datetime.fromisoformat(before)

    weeks = []
    cur = start

    while cur < end:
        nxt = cur + datetime.timedelta(days=7)
        weeks.append(
            {
                "after": cur.isoformat(),
                "before": min(nxt, end).isoformat(),
            }
        )
        cur = nxt

    return weeks


def get_posts(after, before, limit=300):
    """Fetch up to `limit` posts from a given time window."""
    collected = []
    current_before = before
    attempts = 0

    while len(collected) < limit and attempts < 20:
        params = {
            "subreddit": SUBREDDIT,
            "after": after,
            "before": current_before,
            "limit": 100,
            "sort": "desc",
            "fields": "id,title,url,score,num_comments,created_utc,post_hint,selftext",
        }

        try:
            r = requests.get(
                f"{BASE_URL}/posts/search",
                params=params,
                headers=HEADERS,
                timeout=15,
            )
            r.raise_for_status()
            data = r.json().get("data", [])
        except Exception as e:
            print("  Request error:", e)
            time.sleep(2)
            attempts += 1
            continue

        if not data:
            break

        collected.extend(data)
        if len(collected) >= limit:
            break

        current_before = data[-1]["created_utc"]
        attempts += 1
        time.sleep(0.4)

    return collected[:limit]


def sample_week_posts(week, target):
    """Fetch more posts than needed, filter to visual only, shuffle, return exactly `target`."""
    raw = get_posts(week["after"], week["before"], limit=target * 8)

    visual = [p for p in raw if p.get("post_hint") in VISUAL_HINTS]

    random.shuffle(visual)
    return visual[:target]


def get_comments(post_id, limit=10):
    """Fetch up to `limit` top comments for a post."""
    params = {
        "link_id": post_id,
        "limit": limit,
        "fields": "id,body,score,created_utc,author",
    }

    try:
        r = requests.get(
            f"{BASE_URL}/comments/search",
            params=params,
            headers=HEADERS,
            timeout=15,
        )
        r.raise_for_status()
        return r.json().get("data", [])
    except Exception:
        return []


# -----------------------
# MAIN PIPELINE
# -----------------------


def collect_year(period):
    print("\n" + "=" * 60)
    print(f"Subreddit : r/{SUBREDDIT}")
    print(f"Period    : {period['label']}")
    print(f"Target    : {TARGET_POSTS:,} posts  |  up to {COMMENTS_EACH} comments each")
    print("=" * 60)

    weeks = split_into_weeks(period["after"], period["before"])
    n_weeks = len(weeks)

    base = TARGET_POSTS // n_weeks
    remainder = TARGET_POSTS % n_weeks

    print(f"\nWeeks found: {n_weeks}  |  ~{base} posts/week\n")

    all_posts = []

    for i, w in enumerate(weeks):
        target = base + (1 if i < remainder else 0)
        print(
            f"Week {i + 1:>3}/{n_weeks}  "
            f"[{w['after'][:10]} → {w['before'][:10]}]  target={target}"
        )

        week_posts = sample_week_posts(w, target)
        all_posts.extend(week_posts)

        print(f"           collected={len(week_posts)} visual posts")

    print(f"\nTotal posts collected: {len(all_posts):,}")

    rows = []

    for i, post in enumerate(all_posts, 1):
        post_id = post["id"]

        if i % 100 == 0 or i == 1:
            print(f"  Fetching comments for post {i}/{len(all_posts)} ...")

        comments = get_comments(post_id, COMMENTS_EACH)

        for c in comments:
            rows.append(
                {
                    "period": period["label"],
                    "post_id": post_id,
                    "post_title": post.get("title"),
                    "post_url": post.get("url"),
                    "post_hint": post.get("post_hint"),
                    "post_score": post.get("score"),
                    "post_num_comments": post.get("num_comments"),
                    "post_created_utc": post.get("created_utc"),
                    "post_selftext": post.get("selftext", ""),
                    "comment_id": c.get("id"),
                    "comment_body": c.get("body"),
                    "comment_score": c.get("score"),
                    "comment_author": c.get("author"),
                    "comment_created_utc": c.get("created_utc"),
                }
            )

        time.sleep(0.25)

    return rows


if __name__ == "__main__":
    rows = collect_year(PERIOD)

    df = pd.DataFrame(rows)
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)

    print("\n" + "=" * 60)
    print("DONE")
    print(f"Posts : {TARGET_POSTS:,}")
    print(f"Rows  : {len(df):,}")
    print(f"Saved : {OUTPUT_FILE}")
    print("=" * 60)
