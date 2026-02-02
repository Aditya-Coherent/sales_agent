"""
Impact Mapping Agent - Consultant Edition.
Thinks like a consultant to identify which companies need Coherent Market Insights services.
"""

import logging
from typing import List, Dict

from ..schemas import TrendData, ImpactAnalysis, AgentState
from ..tools.llm_tool import LLMTool
from ..config import get_settings, TREND_ROLE_MAPPING, CMI_SERVICES

logger = logging.getLogger(__name__)


class ImpactAgent:
    """
    Agent that thinks like a CONSULTANT to analyze market trends.
    
    For each trend, determines:
    - Which industries are impacted and HOW
    - Which CMI services are most relevant
    - What specific consulting opportunities exist
    - Who are the decision makers who would buy these services
    """
    
    def __init__(self, mock_mode: bool = False):
        """Initialize impact agent with Groq 120B for deep reasoning."""
        self.settings = get_settings()
        self.mock_mode = mock_mode or self.settings.mock_mode
        # Use Groq 120B specifically for impact analysis - best reasoning model
        self.llm_tool = LLMTool(mock_mode=self.mock_mode, force_groq=True)
        logger.info("🧠 Impact Agent initialized with Groq 120B for deep reasoning")
    
    async def analyze_impacts(self, state: AgentState) -> AgentState:
        """
        Analyze impact of all detected trends.
        
        Args:
            state: Current agent state with trends
            
        Returns:
            Updated state with impact analyses
        """
        logger.info("📊 Starting impact analysis...")
        
        if not state.trends:
            logger.warning("No trends to analyze")
            return state
        
        impacts = []
        
        for trend in state.trends:
            try:
                impact = await self._analyze_trend_impact(trend)
                impacts.append(impact)
                logger.info(f"✅ Analyzed impact for: {trend.trend_title[:40]}...")
            except Exception as e:
                logger.warning(f"Failed to analyze impact for trend: {e}")
                # Create basic impact
                impacts.append(self._create_basic_impact(trend))
        
        state.impacts = impacts
        state.current_step = "impacts_analyzed"
        logger.info(f"🎯 Completed {len(impacts)} impact analyses")
        
        return state
    
    async def _analyze_trend_impact(self, trend: TrendData) -> ImpactAnalysis:
        """
        Deep 3-part consultant analysis of a trend using Gemini.
        
        Args:
            trend: Trend data to analyze
            
        Returns:
            ImpactAnalysis with direct, indirect, and additional vertical impacts
        """
        # Build services context
        services_context = "\n".join([
            f"- {svc['name']}: {', '.join(svc['offerings'][:3])}"
            for svc in CMI_SERVICES.values()
        ])
        
        prompt = f"""You are a business development consultant at Coherent Market Insights targeting MID-SIZE INDIAN COMPANIES (50-300 employees).

===== NEWS TO ANALYZE =====
HEADLINE: {trend.trend_title}
DETAILS: {trend.summary}

===== YOUR MISSION =====
Identify where MID-SIZE COMPANIES (not large enterprises, not tiny startups) will STRUGGLE due to this news.
Mid-size companies typically:
- Have limited internal strategy/research teams
- Cannot afford Big 4 consulting (McKinsey, BCG, Deloitte)
- Need actionable market intelligence to compete
- Face resource constraints but ambitious growth goals
- Lack bandwidth to track market changes themselves

Think about SPECIFIC BUSINESS CHALLENGES that create consulting opportunities.

===== EXAMPLE: How to Think Deep =====
NEWS: "India-EU FTA signed - Auto tariffs reduced 30%"

WRONG (too superficial): "Auto industry affected"

RIGHT (mid-size company focused):
- Tier-2 Auto Component Suppliers (50-200 employees): Will face pressure from OEMs to reduce prices OR risk losing contracts to EU suppliers. They need: (1) Cost benchmarking vs EU competitors, (2) Should-cost analysis to negotiate with OEMs, (3) Supplier diversification strategy.
- Mid-size Auto Ancillary Exporters: New opportunity to export to EU, but lack market intelligence on EU regulations, certification requirements, potential buyers. They need: Market entry feasibility study, regulatory compliance mapping.
- Regional Logistics Companies: EU cars need different spare parts distribution. They need: Supply chain reconfiguration study, warehouse location optimization.

===== NOW ANALYZE: {trend.trend_title} =====

Think: What specific mid-size company TYPES will face what SPECIFIC CHALLENGES?

Return this EXACT JSON:

{{
  "direct_impact": [
    "Specific mid-size company type 1 (e.g., 'Tier-2 Oil Field Equipment Suppliers')",
    "Specific mid-size company type 2 (e.g., 'Regional Fuel Distributors')",
    "Specific mid-size company type 3",
    "Specific mid-size company type 4"
  ],
  "direct_impact_reasoning": "For EACH company type above, explain: (1) What specific challenge they face, (2) What decision they need to make, (3) What information/analysis they lack. Be very specific about the business problem, not generic sector impact.",
  
  "indirect_impact": [
    "Mid-size company type affected indirectly 1",
    "Mid-size company type affected indirectly 2", 
    "Mid-size company type affected indirectly 3",
    "Mid-size company type affected indirectly 4"
  ],
  "indirect_impact_reasoning": "Explain the CHAIN: [News] causes [Direct Effect] which creates [Challenge for this mid-size company type]. What specific business decision do they now face? What analysis would help them?",
  
  "additional_verticals": [
    "Non-obvious mid-size company type 1",
    "Non-obvious mid-size company type 2",
    "Non-obvious mid-size company type 3",
    "Non-obvious mid-size company type 4",
    "Non-obvious mid-size company type 5"
  ],
  "additional_verticals_reasoning": "These are mid-size companies most people would NOT think of. Explain the non-obvious connection and what business challenge they face.",
  
  "midsize_pain_points": [
    "Specific pain point 1: e.g., 'Need to renegotiate supplier contracts but lack cost benchmarking data'",
    "Specific pain point 2: e.g., 'Facing margin pressure but dont know competitors pricing strategy'",
    "Specific pain point 3: e.g., 'Want to enter new market segment but lack feasibility analysis'",
    "Specific pain point 4: e.g., 'Board asking for impact assessment but no internal research team'",
    "Specific pain point 5"
  ],
  
  "consulting_projects": [
    "Specific deliverable 1: e.g., 'Cost structure benchmarking for oil field equipment manufacturers'",
    "Specific deliverable 2: e.g., 'Supplier risk assessment and alternative sourcing strategy'",
    "Specific deliverable 3: e.g., 'Market entry feasibility for [specific opportunity]'",
    "Specific deliverable 4: e.g., 'Competitive intelligence on how peers are responding to [this trend]'",
    "Specific deliverable 5"
  ],
  
  "positive_sectors": ["Sectors where mid-size companies will need help navigating change"],
  "negative_sectors": ["Sectors where mid-size companies might cut consulting budgets"],
  "relevant_services": ["Most relevant CMI service 1", "Service 2", "Service 3"],
  "target_roles": ["CEO", "CFO", "VP Strategy", "Director Business Development"],
  "pitch_angle": "One line showing you understand their specific challenge (max 100 chars)",
  "reasoning": "Why would a mid-size company CEO pay for consulting RIGHT NOW based on this news?"
}}

CMI SERVICES WE CAN OFFER:
{services_context}

REMEMBER:
- Focus on MID-SIZE companies (50-300 employees), not Tata/Reliance/Infosys
- Be SPECIFIC about business challenges, not generic sector impacts
- Think about what DECISIONS these companies need to make
- Identify where they LACK INFORMATION that CMI can provide"""

        system_prompt = """You are a business development expert who understands mid-size Indian companies deeply.
You know that mid-size companies (50-300 employees) have unique challenges:
- They're too big to ignore market changes, too small to have in-house research teams
- They compete against both large players AND hungry startups
- They need actionable intelligence, not 200-page reports
- They make decisions fast but need data to back them up
- Their C-suite is accessible and makes buying decisions quickly

Your job is to identify SPECIFIC business challenges where Coherent Market Insights can help.
Always respond with valid JSON only."""

        try:
            result = await self.llm_tool.generate_json(
                prompt=prompt,
                system_prompt=system_prompt
            )
            
            logger.info(f"LLM returned impact analysis with keys: {result.keys()}")
            
            # Enhance target roles based on trend keywords
            target_roles = result.get("target_roles", [])
            if not target_roles:
                target_roles = self._get_roles_from_keywords(trend.keywords)
            
            # Extract the deep mid-size company focused fields
            impact = ImpactAnalysis(
                trend_id=trend.id,
                trend_title=trend.trend_title,
                # Part 1: Direct Impact on Mid-Size Companies
                direct_impact=result.get("direct_impact", []),
                direct_impact_reasoning=result.get("direct_impact_reasoning", ""),
                # Part 2: Indirect Impact
                indirect_impact=result.get("indirect_impact", []),
                indirect_impact_reasoning=result.get("indirect_impact_reasoning", ""),
                # Part 3: Additional Verticals
                additional_verticals=result.get("additional_verticals", []),
                additional_verticals_reasoning=result.get("additional_verticals_reasoning", ""),
                # NEW: Mid-size company pain points
                midsize_pain_points=result.get("midsize_pain_points", []),
                # NEW: Specific consulting projects
                consulting_projects=result.get("consulting_projects", []),
                # Consulting opportunities
                positive_sectors=result.get("positive_sectors", trend.industries_affected),
                negative_sectors=result.get("negative_sectors", []),
                business_opportunities=result.get("consulting_projects", result.get("business_opportunities", [])),  # Use consulting_projects
                target_roles=target_roles,
                relevant_services=result.get("relevant_services", []),
                pitch_angle=result.get("pitch_angle", ""),
                reasoning=result.get("reasoning", "")
            )
            
            logger.info(f"Created ImpactAnalysis with direct_impact: {impact.direct_impact}")
            logger.info(f"Direct impact reasoning: {impact.direct_impact_reasoning[:100]}..." if impact.direct_impact_reasoning else "No direct reasoning")
            
            return impact
            
        except Exception as e:
            logger.warning(f"LLM impact analysis failed: {e}")
            return self._create_basic_impact(trend)
    
    def _create_basic_impact(self, trend: TrendData) -> ImpactAnalysis:
        """Create a basic impact analysis when LLM fails."""
        return ImpactAnalysis(
            trend_id=trend.id,
            trend_title=trend.trend_title,
            # Deep analysis fields with defaults
            direct_impact=trend.industries_affected[:4] if trend.industries_affected else [],
            direct_impact_reasoning=f"These industries are directly mentioned or affected by: {trend.trend_title}. Further analysis with Gemini recommended.",
            indirect_impact=[],
            indirect_impact_reasoning="Indirect impact analysis requires Gemini model. Please check API configuration.",
            additional_verticals=[],
            additional_verticals_reasoning="Additional verticals analysis requires Gemini model. Please check API configuration.",
            # Legacy fields
            positive_sectors=trend.industries_affected,
            negative_sectors=[],
            business_opportunities=[f"Capitalize on {trend.trend_title}"],
            target_roles=self._get_roles_from_keywords(trend.keywords),
            relevant_services=["Market Intelligence", "Industry Analysis"],
            pitch_angle=f"Expert insights on {trend.trend_title[:50]}...",
            reasoning="Market trend with potential impact on Indian industries. Gemini-based deep analysis recommended."
        )
    
    def _get_roles_from_keywords(self, keywords: List[str]) -> List[str]:
        """Determine target roles based on trend keywords."""
        keywords_lower = [k.lower() for k in keywords]
        
        roles = set()
        
        # Check for keyword matches
        keyword_to_type = {
            "ai": "ai",
            "artificial intelligence": "ai",
            "machine learning": "ai",
            "automation": "ai",
            "technology": "technology",
            "digital": "technology",
            "software": "technology",
            "cost": "cost",
            "pricing": "cost",
            "expense": "cost",
            "growth": "growth",
            "expansion": "growth",
            "scale": "growth",
            "funding": "funding",
            "investment": "funding",
            "raise": "funding",
            "regulation": "regulation",
            "compliance": "regulation",
            "policy": "regulation",
            "marketing": "marketing",
            "brand": "marketing",
            "customer": "marketing",
            "supply": "supply_chain",
            "logistics": "supply_chain",
            "manufacturing": "supply_chain"
        }
        
        for keyword in keywords_lower:
            for pattern, trend_type in keyword_to_type.items():
                if pattern in keyword:
                    roles.update(TREND_ROLE_MAPPING.get(trend_type, []))
        
        if not roles:
            roles.update(TREND_ROLE_MAPPING.get("default", []))
        
        return list(roles)[:5]


async def run_impact_agent(state: AgentState) -> AgentState:
    """Wrapper function for LangGraph."""
    agent = ImpactAgent()
    return await agent.analyze_impacts(state)
