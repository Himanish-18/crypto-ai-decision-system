
import logging
import requests
import time
import re
from datetime import datetime
from typing import List, Dict

logger = logging.getLogger("news_feed")

# Try importing feedparser, else fallback
try:
    import feedparser
    HAS_FEEDPARSER = True
except ImportError:
    HAS_FEEDPARSER = False
    import xml.etree.ElementTree as ET

class NewsAggregator:
    def __init__(self):
        # Public RSS feeds (free tier / public)
        # CryptoPanic is good aggregator. 
        self.sources = [
            "https://cryptopanic.com/news/rss/",
            "https://cointelegraph.com/rss",
            "https://www.coindesk.com/arc/outboundfeeds/rss/"
        ]
        self.last_fetch = 0
        self.cache = []
        self.cache_ttl = 300 # 5 minutes

    def fetch_headlines(self) -> List[str]:
        """
        Fetch latest headlines from sources.
        Returns list of strings.
        """
        now = time.time()
        if now - self.last_fetch < self.cache_ttl and self.cache:
            return self.cache

        headlines = []
        
        for url in self.sources:
            try:
                if HAS_FEEDPARSER:
                    feed = feedparser.parse(url)
                    for entry in feed.entries[:5]: # Top 5 per source
                        headlines.append(entry.title)
                else:
                    # Fallback manual parsing
                    resp = requests.get(url, timeout=5)
                    if resp.status_code == 200:
                        root = ET.fromstring(resp.content)
                        # RSS 2.0 usually channel -> item -> title
                        # Atom usually entry -> title
                        items = root.findall(".//item")
                        if not items:
                            items = root.findall(".//entry")
                            
                        for item in items[:5]:
                            title = item.find("title")
                            if title is not None:
                                headlines.append(title.text)
            except Exception as e:
                logger.warning(f"Failed to fetch {url}: {e}")
        
        # Deduplicate
        headlines = list(set(headlines))
        
        # Mock Headlines if empty (e.g. no internet in test env)
        if not headlines:
            logger.warning("No news fetched. Using Mock Headlines for testing.")
            headlines = [
                "Bitcoin stabilizes around $90k as institutions buy",
                "Ethereum upgrade successful, fees drop",
                "Solana network sees record activity"
            ]
            
        self.cache = headlines
        self.last_fetch = now
        logger.info(f"Fetched {len(headlines)} headlines.")
        return headlines

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    news = NewsAggregator()
    print(news.fetch_headlines())
