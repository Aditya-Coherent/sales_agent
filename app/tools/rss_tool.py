"""
RSS Tool for fetching Google News RSS feeds.
Focuses on TODAY's Indian business news - specific events, not generic trends.
"""

import logging
import re
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from urllib.parse import quote_plus
import httpx
import feedparser

from ..config import get_settings, RSS_QUERIES

logger = logging.getLogger(__name__)


class RSSTool:
    """
    RSS feed fetcher for Google News.
    Fetches TODAY's Indian business news - specific events that just happened.
    """
    
    GOOGLE_NEWS_RSS = "https://news.google.com/rss/search"
    
    def __init__(self, mock_mode: bool = False):
        """Initialize RSS tool."""
        self.settings = get_settings()
        self.mock_mode = mock_mode or self.settings.mock_mode
    
    async def fetch_trends(
        self,
        queries: Optional[List[str]] = None,
        max_results: int = 5,
        hours_ago: int = 24
    ) -> List[Dict]:
        """
        Fetch TODAY's news from Google News RSS.
        
        Args:
            queries: List of search queries (defaults to RSS_QUERIES)
            max_results: Maximum results per query
            hours_ago: Only include news from last N hours (default 24)
            
        Returns:
            List of news items with title, summary, link, published date
        """
        if self.mock_mode:
            return self._get_mock_trends()
        
        queries = queries or RSS_QUERIES[:self.settings.max_trends + 2]
        all_items = []
        cutoff_time = datetime.utcnow() - timedelta(hours=hours_ago)
        
        for query in queries:
            try:
                items = await self._fetch_query(query, max_results)
                # Filter for recent news only
                for item in items:
                    if item.get("published"):
                        try:
                            pub_time = datetime.fromisoformat(item["published"].replace("Z", "+00:00"))
                            if pub_time.replace(tzinfo=None) >= cutoff_time:
                                all_items.append(item)
                        except:
                            all_items.append(item)  # Include if can't parse date
                    else:
                        all_items.append(item)
                logger.info(f"Fetched {len(items)} items for query: {query}")
            except Exception as e:
                logger.warning(f"Failed to fetch RSS for '{query}': {e}")
        
        # Sort by recency (newest first)
        all_items.sort(key=lambda x: x.get("published", ""), reverse=True)
        
        # Deduplicate by title
        seen_titles = set()
        unique_items = []
        for item in all_items:
            title_key = item["title"].lower()[:50]
            if title_key not in seen_titles:
                seen_titles.add(title_key)
                unique_items.append(item)
        
        logger.info(f"📰 Found {len(unique_items)} unique news items from last {hours_ago} hours")
        return unique_items[:20]  # Return top 20 unique items
    
    async def _fetch_query(self, query: str, max_results: int) -> List[Dict]:
        """Fetch RSS feed for a specific query."""
        # Build Google News RSS URL with India focus
        encoded_query = quote_plus(f"{query} India")
        url = f"{self.GOOGLE_NEWS_RSS}?q={encoded_query}&hl=en-IN&gl=IN&ceid=IN:en"
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(url)
            response.raise_for_status()
            
            # Parse RSS feed
            feed = feedparser.parse(response.text)
            
            items = []
            for entry in feed.entries[:max_results]:
                item = self._parse_entry(entry, query)
                if item:
                    items.append(item)
            
            return items
    
    def _parse_entry(self, entry: Dict, query: str) -> Optional[Dict]:
        """Parse a single RSS entry."""
        try:
            # Extract clean title (remove source suffix)
            title = entry.get("title", "")
            if " - " in title:
                title = title.rsplit(" - ", 1)[0]
            
            # Extract summary/description
            summary = entry.get("summary", "") or entry.get("description", "")
            # Clean HTML tags
            summary = re.sub(r'<[^>]+>', '', summary)
            summary = summary[:500] if summary else ""
            
            # Parse published date
            published = None
            if entry.get("published_parsed"):
                try:
                    published = datetime(*entry.published_parsed[:6])
                except Exception:
                    pass
            
            return {
                "title": title,
                "summary": summary,
                "link": entry.get("link", ""),
                "source": entry.get("source", {}).get("title", "Google News"),
                "published": published.isoformat() if published else None,
                "query": query
            }
        except Exception as e:
            logger.warning(f"Failed to parse RSS entry: {e}")
            return None
    
    async def fetch_by_topic(self, topic: str, max_results: int = 10) -> List[Dict]:
        """
        Fetch news for a specific topic.
        
        Args:
            topic: Topic to search for
            max_results: Maximum number of results
            
        Returns:
            List of news items
        """
        if self.mock_mode:
            return self._get_mock_trends()[:max_results]
        
        return await self._fetch_query(topic, max_results)
    
    def _get_mock_trends(self) -> List[Dict]:
        """Return mock trends for testing - specific daily news events."""
        # Return 5 DISTINCT news items with unique IDs
        mock_news = [
            {
                "id": "news_001",
                "title": "RBI Mandates New KYC Norms for Digital Lending Apps - 90 Day Deadline",
                "summary": "The Reserve Bank of India today announced stricter KYC requirements for all digital lending platforms. Companies must comply within 90 days or face penalties. This affects over 500 fintech lenders including Lendingkart, Capital Float, and ZestMoney.",
                "link": "https://economictimes.com/news/rbi-kyc-mandate",
                "source": "Economic Times",
                "published": datetime.utcnow().isoformat(),
                "query": "RBI policy announcement"
            },
            {
                "id": "news_002",
                "title": "Swiggy Announces Layoffs of 400 Employees Ahead of IPO",
                "summary": "Food delivery giant Swiggy announced today it will lay off 400 employees as part of cost-cutting measures. The company is focusing on profitability ahead of its planned IPO in Q2 2026. This creates opportunities for HR tech and recruitment firms.",
                "link": "https://moneycontrol.com/news/swiggy-layoffs",
                "source": "Moneycontrol",
                "published": datetime.utcnow().isoformat(),
                "query": "India tech layoffs hiring"
            },
            {
                "id": "news_003",
                "title": "Zepto Raises $200M at $5B Valuation - Expansion to 50 Cities",
                "summary": "Quick commerce startup Zepto closed a $200 million funding round today, valuing the company at $5 billion. Funds will be used to expand dark store network to 50 new cities. This intensifies competition with Blinkit and Instamart.",
                "link": "https://inc42.com/news/zepto-funding",
                "source": "Inc42",
                "published": datetime.utcnow().isoformat(),
                "query": "Indian startup funding announced today"
            },
            {
                "id": "news_004",
                "title": "Cabinet Approves Rs 1.26 Lakh Crore for 3 New Semiconductor Fabs",
                "summary": "The Cabinet today approved setting up of 3 new semiconductor fabrication plants under the India Semiconductor Mission. Tata Electronics and Vedanta are key beneficiaries. Electronics manufacturing sector to see major boost.",
                "link": "https://businessstandard.com/news/semiconductor-approval",
                "source": "Business Standard",
                "published": datetime.utcnow().isoformat(),
                "query": "Indian government scheme launched"
            },
            {
                "id": "news_005",
                "title": "Reliance Jio and NVIDIA Announce AI Cloud Partnership",
                "summary": "Reliance Jio announced a strategic partnership with NVIDIA today to build AI cloud infrastructure in India. Enterprise AI services launching in Q1 2026. This positions Jio against AWS and Azure in the Indian enterprise market.",
                "link": "https://livemint.com/news/jio-nvidia",
                "source": "Mint",
                "published": datetime.utcnow().isoformat(),
                "query": "India business news today"
            }
        ]
        logger.info(f"📰 Returning {len(mock_news)} mock news items")
        return mock_news
