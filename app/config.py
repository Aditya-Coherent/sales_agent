"""
Configuration management for India Trend Lead Agent.
Supports Ollama (local), Gemini (cloud), and Groq (cloud) LLM providers.
"""

import os
from functools import lru_cache
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # LLM Configuration
    use_ollama: bool = Field(default=True, alias="USE_OLLAMA")
    ollama_model: str = Field(default="mistral", alias="OLLAMA_MODEL")
    ollama_base_url: str = Field(default="http://localhost:11434", alias="OLLAMA_BASE_URL")
    gemini_api_key: str = Field(default="", alias="GEMINI_API_KEY")
    
    # Groq Configuration (for 120B reasoning model)
    groq_api_key: str = Field(default="", alias="GROQ_API_KEY")
    groq_model: str = Field(default="openai/gpt-oss-120b", alias="GROQ_MODEL")
    
    # Search APIs
    tavily_api_key: str = Field(default="", alias="TAVILY_API_KEY")
    
    # Email Finder APIs
    apollo_api_key: str = Field(default="", alias="APOLLO_API_KEY")
    hunter_api_key: str = Field(default="", alias="HUNTER_API_KEY")
    
    # Application Settings
    country: str = Field(default="India", alias="COUNTRY")
    max_trends: int = Field(default=3, alias="MAX_TRENDS")
    max_companies_per_trend: int = Field(default=3, alias="MAX_COMPANIES_PER_TREND")
    max_contacts_per_company: int = Field(default=2, alias="MAX_CONTACTS_PER_COMPANY")
    email_confidence_threshold: int = Field(default=70, alias="EMAIL_CONFIDENCE_THRESHOLD")
    mock_mode: bool = Field(default=False, alias="MOCK_MODE")
    
    # Database
    database_url: str = Field(
        default="sqlite+aiosqlite:///./leads.db", 
        alias="DATABASE_URL"
    )
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"
    
    def get_llm_config(self) -> dict:
        """Get LLM configuration based on settings."""
        if self.use_ollama:
            return {
                "provider": "ollama",
                "model": self.ollama_model,
                "base_url": self.ollama_base_url
            }
        else:
            return {
                "provider": "gemini",
                "api_key": self.gemini_api_key,
                "model": "gemini-1.5-flash"
            }


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


# RSS Feed queries for DAILY Indian business news (specific events, not generic trends)
RSS_QUERIES = [
    # Breaking business news
    "India business news today",
    "Indian startup funding announced today",
    "India company acquisition merger",
    "RBI policy announcement",
    "Indian government scheme launched",
    "India regulatory change business",
    "Indian unicorn news",
    "India IPO listing news",
    # Sector-specific breaking news
    "India fintech regulation news",
    "India EV policy announcement",
    "India pharma approval news",
    "India tech layoffs hiring",
]

# Trend type to target role mapping for consulting services
TREND_ROLE_MAPPING = {
    "regulation": ["CEO", "Chief Strategy Officer", "VP Strategy", "Director of Business Development"],
    "policy": ["CEO", "COO", "Chief Strategy Officer", "Director Corporate Strategy"],
    "trade": ["VP Supply Chain", "Procurement Director", "Chief Procurement Officer", "Director Sourcing"],
    "market_shift": ["CMO", "VP Marketing", "Chief Strategy Officer", "Director Market Research"],
    "competition": ["CEO", "Chief Strategy Officer", "VP Business Development", "Director Strategy"],
    "technology": ["CTO", "VP Engineering", "Chief Digital Officer", "Director Innovation"],
    "expansion": ["CEO", "VP Business Development", "Chief Strategy Officer", "Director International"],
    "supply_chain": ["COO", "VP Operations", "Chief Procurement Officer", "Director Supply Chain"],
    "funding": ["CEO", "CFO", "Chief Strategy Officer", "VP Corporate Development"],
    "consumer": ["CMO", "VP Marketing", "Chief Customer Officer", "Director Consumer Insights"],
    "default": ["CEO", "Chief Strategy Officer", "VP Business Development", "Director Strategy"]
}

# Coherent Market Insights Service Catalog
CMI_SERVICES = {
    "procurement_intelligence": {
        "name": "Procurement Intelligence",
        "offerings": [
            "Supplier identification and profiling",
            "Cost structure and should-cost analysis",
            "Commodity and category market analysis",
            "Supply base risk assessment",
            "Procurement process optimization"
        ],
        "keywords": ["supply chain", "procurement", "supplier", "sourcing", "cost", "vendor", "raw material"]
    },
    "market_intelligence": {
        "name": "Market Intelligence",
        "offerings": [
            "Market sizing and segmentation",
            "Market trends and growth forecasts",
            "Regulatory and policy landscape assessment",
            "Market entry and expansion feasibility",
            "Trade Analysis (Export-import analysis)"
        ],
        "keywords": ["market", "growth", "expansion", "entry", "fta", "trade", "export", "import", "demand"]
    },
    "competitive_intelligence": {
        "name": "Competitive Intelligence",
        "offerings": [
            "Competitor profiling and benchmarking",
            "Analysis of competitor strategies",
            "Product and service comparisons",
            "Tracking competitor activities",
            "M&A and partnership tracking"
        ],
        "keywords": ["competitor", "competition", "merger", "acquisition", "market share", "benchmark"]
    },
    "market_monitoring": {
        "name": "Market Monitoring",
        "offerings": [
            "Real-time updates on regulatory changes",
            "Monitoring competitor and supplier activities",
            "Alerts on key market events",
            "Early warning systems for emerging risks"
        ],
        "keywords": ["regulation", "policy", "compliance", "disruption", "risk", "change", "update"]
    },
    "industry_analysis": {
        "name": "Industry Analysis",
        "offerings": [
            "Industry structure and value chain mapping",
            "Key industry drivers and challenges",
            "Regulatory and compliance environment review",
            "Demand and supply dynamics assessment"
        ],
        "keywords": ["industry", "sector", "manufacturing", "pharma", "automotive", "electronics", "chemical"]
    },
    "technology_research": {
        "name": "Technology Research",
        "offerings": [
            "Technology landscape and trends analysis",
            "Assessment of emerging technologies",
            "Technology adoption and impact studies",
            "Patent and intellectual property analysis"
        ],
        "keywords": ["technology", "AI", "automation", "digital", "innovation", "R&D", "tech", "software"]
    },
    "cross_border_expansion": {
        "name": "Cross Border Expansion",
        "offerings": [
            "Market entry strategy and feasibility studies",
            "Regulatory and compliance advisory",
            "Local partner and supplier identification",
            "Go-to-market planning and localization"
        ],
        "keywords": ["expansion", "international", "export", "import", "FTA", "global", "cross-border", "foreign"]
    },
    "consumer_insights": {
        "name": "Consumer Insights",
        "offerings": [
            "Consumer behavior and attitude analysis",
            "Segmentation and persona development",
            "Brand perception and loyalty studies",
            "Customer satisfaction tracking"
        ],
        "keywords": ["consumer", "customer", "brand", "retail", "FMCG", "D2C", "e-commerce"]
    },
    "consulting_advisory": {
        "name": "Consulting and Advisory Services",
        "offerings": [
            "Strategic planning and business transformation",
            "Operational efficiency and process optimization",
            "Technology and digital transformation advisory",
            "Market entry and growth strategy"
        ],
        "keywords": ["strategy", "transformation", "growth", "efficiency", "optimization", "advisory"]
    }
}

# Company size targeting for consulting
TARGET_COMPANY_SIZE = {
    "min_employees": 50,
    "max_employees": 300,
    "size_keywords": ["mid-size", "growing", "emerging", "scaling", "series B", "series C", "established"]
}

# Blacklisted domains (not company domains)
BLACKLISTED_DOMAINS = {
    "linkedin.com", "facebook.com", "twitter.com", "x.com",
    "google.com", "youtube.com", "wikipedia.org",
    "crunchbase.com", "bloomberg.com", "reuters.com",
    "economictimes.com", "moneycontrol.com", "livemint.com",
    "businesstoday.in", "yourstory.com", "inc42.com",
    "github.com", "medium.com", "quora.com"
}

# Company size keywords for classification
COMPANY_SIZE_KEYWORDS = {
    "startup": ["startup", "seed", "early-stage", "series a", "series b", "founded 202"],
    "mid": ["mid-size", "growing", "series c", "series d", "scale-up"],
    "enterprise": ["enterprise", "large", "multinational", "fortune", "listed", "ipo", "public"]
}
