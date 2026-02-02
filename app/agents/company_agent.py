"""
Company Finder Agent.
Finds relevant Indian companies for each trend and impacted sector.
"""

import logging
import hashlib
import re
from typing import List, Dict, Optional

from ..schemas import CompanyData, CompanySize, ImpactAnalysis, AgentState
from ..tools.tavily_tool import TavilyTool
from ..tools.llm_tool import LLMTool
from ..tools.domain_utils import (
    extract_clean_domain, 
    is_valid_company_domain,
    extract_domain_from_company_name,
    extract_domains_from_text
)
from ..config import get_settings, COMPANY_SIZE_KEYWORDS

logger = logging.getLogger(__name__)


class CompanyAgent:
    """
    Agent responsible for finding companies affected by trends.
    
    For each impacted sector:
    - Searches for relevant Indian companies
    - Classifies by size (startup/mid/enterprise)
    - Extracts company websites and domains
    """
    
    def __init__(self, mock_mode: bool = False):
        """Initialize company agent."""
        self.settings = get_settings()
        self.mock_mode = mock_mode or self.settings.mock_mode
        self.tavily_tool = TavilyTool(mock_mode=self.mock_mode)
        self.llm_tool = LLMTool(mock_mode=self.mock_mode)
    
    async def find_companies(self, state: AgentState) -> AgentState:
        """
        Find companies for all trends and impacts.
        
        Args:
            state: Current agent state with trends and impacts
            
        Returns:
            Updated state with companies
        """
        logger.info("🏢 Starting company search...")
        
        if not state.impacts:
            logger.warning("No impacts to process for company search")
            return state
        
        all_companies = []
        max_per_trend = self.settings.max_companies_per_trend
        
        for impact in state.impacts:
            try:
                companies = await self._find_companies_for_impact(impact, max_per_trend)
                all_companies.extend(companies)
                logger.info(f"✅ Found {len(companies)} companies for: {impact.trend_title[:40]}...")
            except Exception as e:
                logger.warning(f"Failed to find companies for impact: {e}")
        
        # Deduplicate by company name
        unique_companies = self._deduplicate_companies(all_companies)
        
        state.companies = unique_companies
        state.current_step = "companies_found"
        logger.info(f"🎯 Found {len(unique_companies)} unique companies")
        
        return state
    
    async def _find_companies_for_impact(
        self,
        impact: ImpactAnalysis,
        limit: int
    ) -> List[CompanyData]:
        """
        Find companies with INTENT SIGNALS from actual news.
        Searches for companies mentioned in news as struggling, expanding, or affected by the trend.
        
        Args:
            impact: Impact analysis with mid-size company types
            limit: Maximum companies to find
            
        Returns:
            List of CompanyData objects with intent signals
        """
        companies = []
        
        # Intent signal keywords to find companies with buying signals
        intent_keywords = [
            "expanding", "struggling", "hiring", "entering market",
            "facing challenges", "seeking partners", "launching",
            "restructuring", "investing in", "scaling", "growing"
        ]
        
        # Build search queries from impact analysis
        search_queries = []
        
        # Priority 1: Search for companies in directly impacted categories with intent
        if hasattr(impact, 'direct_impact') and impact.direct_impact:
            for company_type in impact.direct_impact[:3]:
                # Query 1: Companies struggling/facing challenges
                search_queries.append(f'"{company_type}" India company struggling OR challenges OR facing issues 2024 2025')
                # Query 2: Companies expanding/growing
                search_queries.append(f'"{company_type}" India mid-size company expanding OR growing OR entering market')
        
        # Priority 2: Search based on pain points
        if hasattr(impact, 'midsize_pain_points') and impact.midsize_pain_points:
            for pain_point in impact.midsize_pain_points[:2]:
                # Extract key terms from pain point
                keywords = self._extract_key_terms(pain_point)
                if keywords:
                    search_queries.append(f'India mid-size company {keywords} news')
        
        # Priority 3: Search for companies affected by the trend
        search_queries.append(f'"{impact.trend_title}" affected companies India mid-size')
        
        # Fallback: Direct impact sectors
        if hasattr(impact, 'direct_impact') and impact.direct_impact:
            for sector in impact.direct_impact[:2]:
                search_queries.append(f'top mid-size {sector} companies India 50-300 employees')
        
        logger.info(f"Intent-based search queries: {len(search_queries)}")
        
        for query in search_queries[:5]:  # Limit to 5 searches
            try:
                if self.mock_mode:
                    # In mock mode, generate companies with intent
                    mock_companies = await self._get_mock_companies_with_intent(
                        query, impact, limit
                    )
                    companies.extend(mock_companies)
                else:
                    # Real search for companies with intent signals
                    search_response = await self.tavily_tool.search(query=query, max_results=5)
                    search_results = search_response.get("results", [])
                    
                    # Extract companies from results
                    for result in search_results:
                        try:
                            extracted = await self._extract_companies_with_intent(
                                result, impact, query
                            )
                            companies.extend(extracted)
                        except Exception as e:
                            logger.warning(f"Failed to extract company: {e}")
                
                if len(companies) >= limit:
                    break
                    
            except Exception as e:
                logger.warning(f"Search failed for query '{query[:50]}': {e}")
        
        # Deduplicate
        companies = self._deduplicate_companies(companies)
        
        logger.info(f"Total companies with intent found: {len(companies)}")
        return companies[:limit]
    
    def _extract_key_terms(self, text: str) -> str:
        """Extract key business terms from pain point text."""
        # Remove common words and extract key business terms
        stopwords = {'need', 'to', 'the', 'a', 'an', 'but', 'lack', 'dont', 'know', 'want', 'have', 'is', 'are', 'for', 'of', 'in', 'on', 'with'}
        words = text.lower().replace("'", "").split()
        key_terms = [w for w in words if w not in stopwords and len(w) > 3]
        return ' '.join(key_terms[:4])
    
    async def _get_mock_companies_with_intent(
        self, 
        query: str, 
        impact: ImpactAnalysis, 
        limit: int
    ) -> List[CompanyData]:
        """Generate mock companies with INTENT SIGNALS for testing."""
        import hashlib
        
        # Get pain points and company types for context
        pain_points = getattr(impact, 'midsize_pain_points', [])[:2]
        company_types = getattr(impact, 'direct_impact', [])[:2]
        
        prompt = f"""Find REAL mid-size Indian companies (50-300 employees) that would be affected by this trend.

TREND: {impact.trend_title}
COMPANY TYPES AFFECTED: {', '.join(company_types) if company_types else 'General mid-size companies'}
PAIN POINTS THEY FACE: {', '.join(pain_points) if pain_points else 'Market challenges'}

Return a JSON array of 3-4 REAL Indian companies with:
- company_name: REAL company name (not fictional)
- website: Their actual website
- industry: Their industry
- company_size: "mid" (50-300 employees)
- intent_signal: Why they need consulting NOW (e.g., "Recently announced expansion", "Facing regulatory pressure", "Hiring for new division")
- reason_relevant: Why this trend affects them specifically
- description: One line about what they do

Focus on:
1. Companies actually mentioned in business news
2. Companies showing signs of expansion, struggle, or change
3. Mid-size companies, NOT Tata/Reliance/Infosys
4. Companies with clear consulting needs"""

        try:
            companies_data = await self.llm_tool.generate_list(prompt=prompt)
            
            companies = []
            for item in companies_data[:limit]:
                company_name = item.get("company_name", "")
                if not company_name:
                    continue
                    
                website = item.get("website", "")
                domain = ""
                if website:
                    domain = extract_clean_domain(website)
                if not domain:
                    domain = extract_domain_from_company_name(company_name)
                
                company_id = hashlib.md5(company_name.encode()).hexdigest()[:12]
                
                # Build reason with intent signal
                intent = item.get("intent_signal", "")
                reason = item.get("reason_relevant", f"Affected by {impact.trend_title[:30]}")
                full_reason = f"{intent}. {reason}" if intent else reason
                
                companies.append(CompanyData(
                    id=company_id,
                    company_name=company_name,
                    company_size=CompanySize(item.get("company_size", "mid")),
                    industry=item.get("industry", "Technology"),
                    website=website or (f"https://{domain}" if domain else ""),
                    domain=domain or "",
                    description=item.get("description", ""),
                    reason_relevant=full_reason,
                    trend_id=impact.trend_id
                ))
            
            logger.info(f"Generated {len(companies)} companies with intent for trend")
            return companies
            
        except Exception as e:
            logger.error(f"Failed to generate mock companies: {e}")
            return []
    
    async def _get_mock_companies(self, sector: str, trend_id: str, limit: int) -> List[CompanyData]:
        """Generate mock companies for testing (legacy, kept for compatibility)."""
        import hashlib
        
        # Use LLM tool to get mock companies
        prompt = f"Find companies in {sector} sector in India"
        companies_data = await self.llm_tool.generate_list(prompt=prompt)
        
        companies = []
        for item in companies_data[:limit]:
            company_name = item.get("company_name", "")
            if not company_name:
                continue
                
            website = item.get("website", "")
            domain = ""
            if website:
                domain = extract_clean_domain(website)
            if not domain:
                domain = extract_domain_from_company_name(company_name)
            
            company_id = hashlib.md5(company_name.encode()).hexdigest()[:12]
            
            companies.append(CompanyData(
                id=company_id,
                company_name=company_name,
                company_size=CompanySize(item.get("company_size", "mid")),
                industry=sector,
                website=website or (f"https://{domain}" if domain else ""),
                domain=domain or "",
                description=item.get("description", ""),
                reason_relevant=item.get("reason_relevant", f"Active in {sector}"),
                trend_id=trend_id
            ))
        
        logger.info(f"Generated {len(companies)} mock companies for {sector}")
        return companies
    
    async def _extract_companies_with_intent(
        self,
        search_result: Dict,
        impact: ImpactAnalysis,
        query: str
    ) -> List[CompanyData]:
        """
        Extract companies with INTENT SIGNALS from search results.
        Focuses on finding companies mentioned in news with buying signals.
        """
        snippet = search_result.get("content", search_result.get("snippet", ""))
        source_url = search_result.get("url", search_result.get("source_url", ""))
        title = search_result.get("title", "")
        
        # Use LLM to extract companies with intent
        prompt = f"""Extract mid-size Indian companies (50-300 employees) from this news/article.

ARTICLE TITLE: {title}
ARTICLE TEXT: {snippet}
SOURCE: {source_url}

CONTEXT - We are looking for companies affected by: {impact.trend_title}

For each company mentioned, extract as JSON array:
- company_name: Official company name
- website: Company website if found
- industry: Their industry
- intent_signal: What action/change they are taking (e.g., "expanding operations", "facing regulatory pressure", "hiring for new division", "restructuring", "seeking partnerships")
- reason_relevant: Why this company would need consulting services based on the news
- description: What the company does

RULES:
1. Only include REAL companies clearly mentioned by name
2. Focus on mid-size companies (NOT Tata, Reliance, Infosys, Wipro)
3. Must have clear INTENT SIGNAL (action they are taking or challenge they face)
4. Return empty array [] if no qualifying companies found

Return JSON array only."""

        try:
            result = await self.llm_tool.generate_list(prompt=prompt)
            
            companies = []
            for item in result:
                company_name = item.get("company_name", "")
                if not company_name:
                    continue
                
                # Skip large enterprises
                large_companies = ['tata', 'reliance', 'infosys', 'wipro', 'hcl', 'hdfc', 'icici', 'bajaj', 'mahindra', 'adani', 'vedanta']
                if any(lc in company_name.lower() for lc in large_companies):
                    continue
                
                website = item.get("website", "")
                domain = ""
                if website:
                    domain = extract_clean_domain(website)
                if not domain:
                    domain = await self._find_company_domain(company_name)
                if not domain:
                    domain = extract_domain_from_company_name(company_name)
                
                if domain and not is_valid_company_domain(domain):
                    domain = ""
                
                company_id = hashlib.md5(company_name.encode()).hexdigest()[:12]
                
                # Build reason with intent
                intent = item.get("intent_signal", "")
                reason = item.get("reason_relevant", "")
                full_reason = f"📌 {intent}. {reason}" if intent else reason
                
                companies.append(CompanyData(
                    id=company_id,
                    company_name=company_name,
                    company_size=CompanySize.MID,
                    industry=item.get("industry", "Technology"),
                    website=website or (f"https://{domain}" if domain else ""),
                    domain=domain or "",
                    description=item.get("description", ""),
                    reason_relevant=full_reason,
                    trend_id=impact.trend_id
                ))
            
            if companies:
                logger.info(f"Extracted {len(companies)} companies with intent from: {title[:50]}")
            return companies
            
        except Exception as e:
            logger.warning(f"LLM extraction failed: {e}")
            return []
    
    async def _extract_companies_from_result(
        self,
        search_result: Dict,
        sector: str,
        trend_id: str
    ) -> List[CompanyData]:
        """
        Extract company information from a search result (legacy method).
        """
        snippet = search_result.get("snippet", "")
        source_url = search_result.get("source_url", "")
        
        prompt = f"""Extract company information from this text about {sector} companies in India.

TEXT: {snippet}
SOURCE URL: {source_url}

For each company mentioned, extract as JSON array with:
- company_name: Official company name
- website: Company website URL if mentioned
- description: One-sentence description
- reason_relevant: Why relevant to {sector}

Only include Indian companies. Return JSON array or empty []."""

        try:
            result = await self.llm_tool.generate_list(prompt=prompt)
            
            companies = []
            for item in result:
                if not item.get("company_name"):
                    continue
                
                company_name = item.get("company_name", "")
                website = item.get("website", "")
                
                domain = ""
                if website:
                    domain = extract_clean_domain(website)
                if not domain:
                    domain = await self._find_company_domain(company_name)
                if not domain:
                    domain = extract_domain_from_company_name(company_name)
                if domain and not is_valid_company_domain(domain):
                    domain = ""
                
                company_id = hashlib.md5(company_name.encode()).hexdigest()[:12]
                size = self._classify_company_size(snippet, company_name)
                
                companies.append(CompanyData(
                    id=company_id,
                    company_name=company_name,
                    company_size=size,
                    industry=sector,
                    website=website or (f"https://{domain}" if domain else ""),
                    domain=domain or "",
                    description=item.get("description", ""),
                    reason_relevant=item.get("reason_relevant", f"Active in {sector} sector"),
                    trend_id=trend_id
                ))
            
            return companies
            
        except Exception as e:
            logger.warning(f"LLM extraction failed: {e}")
            return []
    
    async def _find_company_domain(self, company_name: str) -> Optional[str]:
        """Find company domain via search."""
        try:
            domain = await self.tavily_tool.find_company_domain(company_name)
            return domain
        except Exception:
            return None
    
    def _classify_company_size(self, text: str, company_name: str) -> CompanySize:
        """Classify company size based on context."""
        text_lower = text.lower() + " " + company_name.lower()
        
        for size, keywords in COMPANY_SIZE_KEYWORDS.items():
            for keyword in keywords:
                if keyword in text_lower:
                    return CompanySize(size)
        
        # Default to mid-size
        return CompanySize.MID
    
    def _deduplicate_companies(self, companies: List[CompanyData]) -> List[CompanyData]:
        """Remove duplicate companies by name."""
        seen = set()
        unique = []
        
        for company in companies:
            key = company.company_name.lower().strip()
            if key not in seen:
                seen.add(key)
                unique.append(company)
        
        return unique


async def run_company_agent(state: AgentState) -> AgentState:
    """Wrapper function for LangGraph."""
    agent = CompanyAgent()
    return await agent.find_companies(state)
