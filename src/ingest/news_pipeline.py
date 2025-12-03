"""
Optimized & Commented Institutional News Ingestion Pipeline
Sources:
 - CryptoPanic (API)
 - RSS crypto feeds (Google News / Cointelegraph / Coindesk)
 
Removed:
 - Pushshift (permanently shut down → always 403 errors)

Adds:
 - Deduplication
 - Sentiment (VADER + TextBlob)
 - Credibility scoring (vectorized)
 - Checkpoints (safe resume)
 - Optimized loops & sessions
"""

import os
import time
import json
import logging
from typing import List, Optional, Dict, Any
from datetime import datetime, timezone
from urllib.parse import urlparse

import requests
import pandas as pd
import feedparser
from tqdm import tqdm
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob

# ---------------------------
# Logging Setup
# ---------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("news_pipeline")

# ---------------------------
# Directories
# ---------------------------
CHECKPOINT_DIR = ".checkpoints_news"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

def save_checkpoint(name: str, data: Any):
    """Save progress to resume later (safe for long jobs)."""
    try:
        with open(f"{CHECKPOINT_DIR}/{name}.json", "w") as f:
            json.dump(data, f)
        log.info(f"💾 Checkpoint saved: {name}")
    except Exception as e:
        log.error(f"Failed to save checkpoint {name}: {e}")

def load_checkpoint(name: str) -> Optional[Any]:
    """Load previous progress if exists."""
    fp = f"{CHECKPOINT_DIR}/{name}.json"
    if os.path.exists(fp):
        log.info(f"⏪ Loading checkpoint: {name}")
        with open(fp) as f:
            return json.load(f)
    return None

# ---------------------------
# Helpers
# ---------------------------
def normalize_date(dt: Any) -> Optional[datetime]:
    """Safely parse timestamps from RSS/CryptoPanic."""
    if dt is None or dt == "":
        return None

    try:
        ts = pd.to_datetime(dt, utc=True)
        return ts.to_pydatetime()
    except Exception:
        try:
            return datetime.utcfromtimestamp(float(dt)).replace(tzinfo=timezone.utc)
        except Exception:
            return None

def domain(url: str) -> str:
    """Extract domain name from a URL."""
    try:
        return urlparse(url).netloc or "unknown"
    except:
        return "unknown"

# ---------------------------
# Credibility Scoring
# ---------------------------
TRUST_SOURCES = {
    "coindesk.com": 1.0,
    "cointelegraph.com": 1.0,
    "reuters.com": 1.0,
    "bloomberg.com": 1.0,
    "theblock.co": 0.95,
    "decrypt.co": 0.9,
}

def apply_vectorized_credibility(df: pd.DataFrame) -> pd.Series:
    """Vectorized credibility score (fast)."""
    log.info("Applying credibility scores...")

    credibility = pd.Series(0.5, index=df.index)

    # Trusted domains boost score
    credibility += (df["domain"].map(TRUST_SOURCES).fillna(0.5) - 0.5)

    # Text length signals
    text_len = (df["title"].fillna("") + " " + df["text"].fillna("")).str.len()
    credibility[text_len < 50] -= 0.2
    credibility[text_len > 200] += 0.1

    return credibility.clip(0, 1)

# ---------------------------
# Sentiment
# ---------------------------
vader = SentimentIntensityAnalyzer()

def sentiment(text: str):
    """Compute sentiment using VADER + TextBlob."""
    if not text:
        return 0.0, 0.0, 0.0

    try:
        v = vader.polarity_scores(text)["compound"]
        t = TextBlob(text)
        return v, float(t.polarity), float(t.subjectivity)
    except:
        return 0.0, 0.0, 0.0

# ---------------------------
# CryptoPanic Fetcher
# ---------------------------
def fetch_cryptopanic(session, api_key, max_items=50000):
    if not api_key:
        log.warning("⚠ No CryptoPanic key → Skipping.")
        return []

    log.info("🔵 Fetching CryptoPanic news...")
    items = []
    page = 1

    with tqdm(total=max_items, desc="CryptoPanic") as bar:
        while len(items) < max_items:
            try:
                r = session.get(
                    "https://cryptopanic.com/api/v1/posts/",
                    params={
                        "auth_token": api_key,
                        "public": True,
                        "kind": "news",
                        "page": page,
                    },
                    timeout=15,
                )

                if r.status_code != 200:
                    log.warning(f"CryptoPanic HTTP {r.status_code}, retrying...")
                    time.sleep(2)
                    continue
            except Exception as e:
                log.error(f"Error: {e}")
                time.sleep(2)
                continue

            results = r.json().get("results", [])
            if not results:
                break

            for it in results:
                items.append({
                    "title": it.get("title", ""),
                    "text": it.get("body", ""),
                    "timestamp": normalize_date(it.get("published_at")),
                    "url": it.get("url", ""),
                    "domain": domain(it.get("url", "")),
                    "source": "cryptopanic",
                    "origin": "cryptopanic"
                })

            bar.update(len(results))
            page += 1

    return items

# ---------------------------
# RSS Fetcher (Main Free Source)
# ---------------------------
def fetch_rss(session, feeds, limit=1000):
    log.info("🔵 Fetching RSS feeds…")

    articles = []

    for feed in tqdm(feeds, desc="RSS"):
        try:
            r = session.get(feed, timeout=10)
            f = feedparser.parse(r.content)

            for ent in f.entries[:limit]:
                articles.append({
                    "title": ent.get("title", ""),
                    "text": ent.get("summary", ent.get("description", "")),
                    "timestamp": normalize_date(ent.get("published")),
                    "url": ent.get("link", ""),
                    "domain": domain(ent.get("link", "")),
                    "source": f.feed.get("title", "rss"),
                    "origin": "rss",
                })

        except Exception as e:
            log.error(f"RSS error ({feed}): {e}")

    return articles

# ---------------------------
# Main Pipeline
# ---------------------------
def run_pipeline(
    cryptopanic_key=None,
    rss_feeds=None,
    max_items=200000,
    out_file="data/raw/news_combined.parquet"
):
    rss_feeds = rss_feeds or [
        "https://news.google.com/rss/search?q=bitcoin&hl=en-US&gl=US&ceid=US:en",
        "https://news.google.com/rss/search?q=ethereum&hl=en-US&gl=US&ceid=US:en",
        "https://www.coindesk.com/arc/outboundfeeds/rss/",
        "https://cointelegraph.com/rss",
        "https://cryptoslate.com/feed/",
    ]

    with requests.Session() as session:
        session.headers.update({"User-Agent": "NewsPipeline/1.0"})

        # Fetch CryptoPanic (optional)
        cp = load_checkpoint("cryptopanic") or fetch_cryptopanic(session, cryptopanic_key, max_items)
        save_checkpoint("cryptopanic", cp)

        # Fetch RSS (main free data source)
        rs = load_checkpoint("rss") or fetch_rss(session, rss_feeds, limit=5000)
        save_checkpoint("rss", rs)

    df = pd.DataFrame(cp + rs)
    log.info(f"Fetched {len(df)} raw rows")

    if df.empty:
        log.warning("No data fetched.")
        return

    df = df.dropna(subset=["timestamp"])

    # Deduplicate by title + domain
    df["dedupe_key"] = df["title"].str.lower().str.strip() + "||" + df["domain"]
    df = df.sort_values("timestamp").drop_duplicates("dedupe_key", keep="last")

    log.info(f"After dedupe: {len(df)} rows")

    # Sentiment
    vader_scores = []
    tb_pol = []
    tb_subj = []

    for t in tqdm((df["title"] + " " + df["text"]).fillna(""), desc="Sentiment"):
        v, p, s = sentiment(t)
        vader_scores.append(v)
        tb_pol.append(p)
        tb_subj.append(s)

    df["vader_compound"] = vader_scores
    df["textblob_polarity"] = tb_pol
    df["textblob_subjectivity"] = tb_subj

    # Credibility
    df["credibility"] = apply_vectorized_credibility(df)

    final_cols = [
        "timestamp", "title", "text", "url", "domain", "source", "origin",
        "vader_compound", "textblob_polarity", "textblob_subjectivity", "credibility"
    ]

    df = df[final_cols]

    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    df.to_parquet(out_file, index=False)
    df.to_csv(out_file.replace(".parquet", ".csv"), index=False)

    log.info(f"✅ DONE → Saved {len(df)} rows")
    return df


# ---------------------------
# CLI
# ---------------------------
if __name__ == "__main__":
    api_key = os.getenv("CRYPTOPANIC_API_KEY")
    if not api_key:
        log.warning("CryptoPanic disabled (no API key)")

    run_pipeline(
        cryptopanic_key=api_key,
        max_items=150000,
        out_file="data/raw/news_combined.parquet"
    )
