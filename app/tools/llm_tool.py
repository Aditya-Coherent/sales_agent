"""
LLM Tool with Ollama (local), Gemini (cloud), and Groq (cloud) support.
Automatic fallback chain: Groq -> Gemini -> Ollama.
Groq's 120B model is used for deep reasoning tasks.
"""

import logging
import json
import re
from typing import Optional, Dict, Any, List
import httpx

# Try new google-genai package first, fall back to old one
try:
    from google import genai
    from google.genai import types
    NEW_GENAI = True
except ImportError:
    try:
        import google.generativeai as genai
        NEW_GENAI = False
    except ImportError:
        genai = None
        NEW_GENAI = False

# Try Groq
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    Groq = None
    GROQ_AVAILABLE = False

from ..config import get_settings

logger = logging.getLogger(__name__)


class LLMTool:
    """
    LLM wrapper supporting Groq, Gemini, and Ollama with automatic fallback.
    
    Priority for deep reasoning (force_groq=True):
    1. Groq 120B (best reasoning, cloud)
    2. Gemini (fallback)
    3. Ollama (local fallback)
    
    Priority for normal tasks:
    1. Ollama (local, free) if USE_OLLAMA=true
    2. Gemini (cloud)
    """
    
    def __init__(self, mock_mode: bool = False, force_gemini: bool = False, force_groq: bool = False):
        """Initialize LLM tool.
        
        Args:
            mock_mode: Use mock responses instead of real API calls
            force_gemini: Always use Gemini instead of Ollama (for better reasoning)
            force_groq: Always use Groq 120B for deep reasoning (recommended for impact analysis)
        """
        self.settings = get_settings()
        self.mock_mode = mock_mode or self.settings.mock_mode
        self.force_gemini = force_gemini
        self.force_groq = force_groq
        self._gemini_configured = False
        self._gemini_client = None
        self._groq_client = None
        self._groq_configured = False
        
        # Configure Groq if API key available (priority for reasoning)
        if self.settings.groq_api_key and GROQ_AVAILABLE:
            try:
                self._groq_client = Groq(api_key=self.settings.groq_api_key)
                self._groq_configured = True
                logger.info(f"Groq API configured with model: {self.settings.groq_model}")
            except Exception as e:
                logger.warning(f"Failed to configure Groq: {e}")
        
        # Configure Gemini if API key available
        if self.settings.gemini_api_key and genai is not None:
            try:
                if NEW_GENAI:
                    # New google-genai package
                    self._gemini_client = genai.Client(api_key=self.settings.gemini_api_key)
                else:
                    # Old google-generativeai package
                    genai.configure(api_key=self.settings.gemini_api_key)
                self._gemini_configured = True
                logger.info("Gemini API configured successfully")
            except Exception as e:
                logger.warning(f"Failed to configure Gemini: {e}")
    
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        json_mode: bool = False
    ) -> str:
        """
        Generate text using LLM.
        
        Args:
            prompt: User prompt
            system_prompt: Optional system instructions
            temperature: Creativity level (0-1)
            max_tokens: Maximum response length
            json_mode: If True, expect JSON response
            
        Returns:
            Generated text response
        """
        if self.mock_mode:
            return self._get_mock_response(prompt, json_mode)
        
        # Priority 1: Groq 120B for deep reasoning (best for impact analysis)
        if self.force_groq and self._groq_configured:
            logger.info("🧠 Using Groq 120B for deep reasoning (force_groq=True)")
            try:
                response = await self._call_groq(
                    prompt, system_prompt, temperature, max_tokens
                )
                if response:
                    return response
            except Exception as e:
                logger.warning(f"Groq failed: {e}, falling back to Gemini")
        
        # Priority 2: Gemini if forcing or Groq failed
        if (self.force_gemini or self.force_groq) and self._gemini_configured:
            logger.info("Using Gemini for reasoning")
            try:
                response = await self._call_gemini(
                    prompt, system_prompt, temperature, max_tokens
                )
                if response:
                    return response
            except Exception as e:
                logger.error(f"Gemini failed: {e}")
        
        # Priority 3: Ollama for normal tasks
        if self.settings.use_ollama and not self.force_gemini and not self.force_groq:
            try:
                response = await self._call_ollama(
                    prompt, system_prompt, temperature, max_tokens
                )
                if response:
                    return response
            except Exception as e:
                logger.warning(f"Ollama failed, falling back to cloud: {e}")
        
        # Final fallback: Try Gemini
        if self._gemini_configured:
            try:
                response = await self._call_gemini(
                    prompt, system_prompt, temperature, max_tokens
                )
                if response:
                    return response
            except Exception as e:
                logger.error(f"Gemini failed: {e}")
        
        raise RuntimeError("All LLM providers failed (Groq, Gemini, Ollama)")
    
    async def _call_ollama(
        self,
        prompt: str,
        system_prompt: Optional[str],
        temperature: float,
        max_tokens: int
    ) -> Optional[str]:
        """Call Ollama API."""
        url = f"{self.settings.ollama_base_url}/api/generate"
        
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"
        
        payload = {
            "model": self.settings.ollama_model,
            "prompt": full_prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens
            }
        }
        
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            
            data = response.json()
            return data.get("response", "")
    
    async def _call_gemini(
        self,
        prompt: str,
        system_prompt: Optional[str],
        temperature: float,
        max_tokens: int
    ) -> Optional[str]:
        """Call Gemini API."""
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"
        
        if NEW_GENAI and self._gemini_client:
            # New google-genai package
            response = self._gemini_client.models.generate_content(
                model='gemini-2.0-flash-exp',
                contents=full_prompt,
                config=types.GenerateContentConfig(
                    temperature=temperature,
                    max_output_tokens=max_tokens
                )
            )
            return response.text
        else:
            # Old google-generativeai package
            model = genai.GenerativeModel('gemini-1.5-flash')
            
            generation_config = genai.GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_tokens
            )
            
            response = model.generate_content(
                full_prompt,
                generation_config=generation_config
            )
            
            return response.text
    
    async def _call_groq(
        self,
        prompt: str,
        system_prompt: Optional[str],
        temperature: float,
        max_tokens: int
    ) -> Optional[str]:
        """
        Call Groq API with high-quality models.
        Supports llama-3.3-70b-versatile (best for JSON) and openai/gpt-oss-120b (reasoning).
        """
        if not self._groq_client:
            return None
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        try:
            model = self.settings.groq_model
            logger.info(f"🧠 Calling Groq model: {model}")
            
            # Build API call params based on model type
            call_params = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_completion_tokens": max_tokens,
                "top_p": 1,
                "stream": False,
            }
            
            # Only add reasoning_effort for 120B model
            if "120b" in model.lower() or "oss" in model.lower():
                call_params["reasoning_effort"] = "high"
                logger.info("Using reasoning_effort=high for 120B model")
            
            completion = self._groq_client.chat.completions.create(**call_params)
            
            # Get the response content
            response_content = completion.choices[0].message.content or ""
            
            # For 120B model, if content is empty but reasoning exists, use reasoning
            if not response_content and hasattr(completion.choices[0].message, 'reasoning'):
                reasoning = completion.choices[0].message.reasoning
                if reasoning:
                    logger.info("Content empty, using reasoning field as response")
                    response_content = reasoning
            
            # Log reasoning if available (for debugging)
            if hasattr(completion.choices[0].message, 'reasoning') and completion.choices[0].message.reasoning:
                reasoning = completion.choices[0].message.reasoning
                logger.info(f"🧠 Groq reasoning trace: {reasoning[:150]}...")
            
            logger.info(f"✅ Groq response length: {len(response_content)} chars")
            return response_content
            
        except Exception as e:
            logger.error(f"Groq API call failed: {e}")
            raise
    
    async def generate_json(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        schema_hint: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate structured JSON response.
        
        Args:
            prompt: User prompt
            system_prompt: Optional system instructions
            schema_hint: Optional JSON schema description
            
        Returns:
            Parsed JSON dictionary
        """
        json_instruction = """
You must respond with valid JSON only. No markdown, no explanation, just the JSON object.
Do not wrap the response in ```json``` code blocks.
"""
        if schema_hint:
            json_instruction += f"\nExpected format:\n{schema_hint}"
        
        full_system = (system_prompt or "") + "\n" + json_instruction
        
        response = await self.generate(
            prompt=prompt,
            system_prompt=full_system.strip(),
            temperature=0.3,  # Lower temperature for structured output
            json_mode=True
        )
        
        return self._parse_json_response(response)
    
    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """Parse JSON from LLM response, handling common issues."""
        # Remove markdown code blocks if present
        cleaned = response.strip()
        if cleaned.startswith("```"):
            # Remove opening fence
            cleaned = re.sub(r'^```(?:json)?\n?', '', cleaned)
            # Remove closing fence
            cleaned = re.sub(r'\n?```$', '', cleaned)
        
        # Check if response starts with [ (array) or { (object)
        cleaned = cleaned.strip()
        
        # Try to find JSON array FIRST (important for company lists)
        if cleaned.startswith('['):
            array_match = re.search(r'\[[\s\S]*\]', cleaned)
            if array_match:
                cleaned = array_match.group()
        # Then try JSON object
        elif cleaned.startswith('{'):
            json_match = re.search(r'\{[\s\S]*\}', cleaned)
            if json_match:
                cleaned = json_match.group()
        else:
            # Fallback: try array then object
            array_match = re.search(r'\[[\s\S]*\]', cleaned)
            if array_match:
                cleaned = array_match.group()
            else:
                json_match = re.search(r'\{[\s\S]*\}', cleaned)
                if json_match:
                    cleaned = json_match.group()
        
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON: {e}\nResponse: {cleaned[:500]}")
            return {"error": "Failed to parse JSON", "raw": cleaned[:500]}
    
    async def generate_list(
        self,
        prompt: str,
        system_prompt: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Generate a list of JSON objects."""
        json_instruction = """
You must respond with a valid JSON array only. No markdown, no explanation.
Each item in the array should be a JSON object.
Do not wrap the response in ```json``` code blocks.
"""
        full_system = (system_prompt or "") + "\n" + json_instruction
        
        response = await self.generate(
            prompt=prompt,
            system_prompt=full_system.strip(),
            temperature=0.3,
            json_mode=True
        )
        
        result = self._parse_json_response(response)
        
        # If we got a dict with a list inside, extract it
        if isinstance(result, dict):
            for key in ["items", "results", "data", "companies", "contacts", "trends"]:
                if key in result and isinstance(result[key], list):
                    return result[key]
            return [result]
        
        return result if isinstance(result, list) else [result]
    
    def _get_mock_response(self, prompt: str, json_mode: bool) -> str:
        """Return mock response for testing."""
        import random
        import hashlib
        
        # Generate a unique response based on prompt content
        prompt_hash = int(hashlib.md5(prompt.encode()).hexdigest()[:8], 16) % 5
        
        prompt_lower = prompt.lower()
        
        if json_mode or "json" in prompt_lower:
            # Check for companies FIRST (before trend check, because company prompts may contain "trend")
            if ("compan" in prompt_lower and "find" in prompt_lower) or "real" in prompt_lower or "extract" in prompt_lower:
                # Return company data - handled in company section below
                return self._get_mock_company_response(prompt_lower)
            elif "trend" in prompt_lower or "news" in prompt_lower:
                mock_trends = [
                    {
                        "trend_title": "RBI Mandates Stricter KYC for Digital Lenders",
                        "summary": "Reserve Bank of India announced new KYC requirements affecting 500+ fintech lenders. Companies must comply within 90 days.",
                        "severity": "high",
                        "industries_affected": ["Fintech", "Digital Lending", "NBFC", "Banking"],
                        "keywords": ["RBI", "KYC", "digital lending", "compliance", "fintech regulation"],
                        "trend_type": "regulation",
                        "urgency": "Immediate compliance required - 90 day deadline"
                    },
                    {
                        "trend_title": "Zepto Raises $200M, Valued at $5B",
                        "summary": "Quick commerce startup Zepto closed massive funding round, plans expansion to 50 new cities. Competitor pressure intensifies.",
                        "severity": "high",
                        "industries_affected": ["Quick Commerce", "E-commerce", "Logistics", "Retail"],
                        "keywords": ["Zepto", "quick commerce", "funding", "dark stores", "Blinkit", "Instamart"],
                        "trend_type": "funding",
                        "urgency": "Competitors need to respond to market pressure"
                    },
                    {
                        "trend_title": "Swiggy Announces 400 Employee Layoffs",
                        "summary": "Food delivery giant Swiggy cuts 400 jobs in restructuring ahead of IPO. Focus shifts to profitability over growth.",
                        "severity": "medium",
                        "industries_affected": ["Food Tech", "Gig Economy", "HR Tech", "Recruitment"],
                        "keywords": ["Swiggy", "layoffs", "IPO", "restructuring", "food delivery"],
                        "trend_type": "layoffs",
                        "urgency": "Talent available in market, HR solutions needed"
                    },
                    {
                        "trend_title": "Government Approves 3 New Semiconductor Fabs",
                        "summary": "Cabinet clears Rs 1.26 lakh crore investment for semiconductor manufacturing. Major opportunity for electronics ecosystem.",
                        "severity": "high",
                        "industries_affected": ["Semiconductors", "Electronics", "Manufacturing", "IT Hardware"],
                        "keywords": ["semiconductor", "PLI scheme", "electronics manufacturing", "chip fab"],
                        "trend_type": "policy",
                        "urgency": "First-mover advantage in emerging ecosystem"
                    },
                    {
                        "trend_title": "Reliance Jio Partners with NVIDIA for AI Cloud",
                        "summary": "Jio and NVIDIA announce strategic partnership for enterprise AI infrastructure in India. New AI cloud services launching.",
                        "severity": "high",
                        "industries_affected": ["Cloud Computing", "AI/ML", "Enterprise IT", "Data Centers"],
                        "keywords": ["Jio", "NVIDIA", "AI cloud", "enterprise AI", "GPU computing"],
                        "trend_type": "partnership",
                        "urgency": "Enterprises need AI strategy now"
                    }
                ]
                return json.dumps(mock_trends[prompt_hash])
            elif "impact" in prompt.lower() or "consultant" in prompt.lower() or "direct" in prompt.lower():
                mock_impacts = [
                    {
                        "direct_impact": ["Digital Lending NBFCs", "Fintech Payment Apps", "P2P Lending Platforms", "Buy Now Pay Later Companies"],
                        "direct_impact_reasoning": "RBI's new KYC norms directly mandate these companies to overhaul their customer verification processes. They face a 90-day compliance deadline with penalties for non-compliance. Most mid-size lenders lack dedicated compliance teams.",
                        "indirect_impact": ["RegTech Software Providers", "Identity Verification Services", "Compliance Consulting Firms", "Customer Onboarding Platforms"],
                        "indirect_impact_reasoning": "As lenders scramble to comply, they'll need technology and consulting support. RegTech demand will surge. KYC verification vendors will see increased volumes. Compliance consultants will be hired for gap assessments.",
                        "additional_verticals": ["Banking Software Vendors", "Data Analytics Firms", "Cybersecurity Companies", "Legal Advisory Firms", "Document Management Solutions", "API Integration Specialists"],
                        "additional_verticals_reasoning": "The KYC overhaul requires system upgrades (software vendors), data handling changes (analytics), security enhancements (cybersecurity), legal review (law firms), and document workflows (DMS). API specialists needed for Aadhaar/PAN integration.",
                        "positive_sectors": ["Fintech Lenders", "RegTech", "Compliance Consulting", "Identity Verification"],
                        "negative_sectors": ["Unorganized Moneylenders"],
                        "business_opportunities": [
                            "KYC compliance gap assessment for mid-size NBFCs",
                            "Regulatory landscape mapping for fintech lenders",
                            "Cost-benefit analysis of compliance technology options",
                            "Benchmarking study of KYC best practices",
                            "Vendor evaluation for identity verification solutions"
                        ],
                        "relevant_services": ["Market Monitoring", "Consulting and Advisory Services", "Technology Research"],
                        "target_roles": ["CEO", "Chief Compliance Officer", "VP Operations", "Director Risk Management"],
                        "pitch_angle": "Navigate RBI's KYC mandate with expert compliance guidance",
                        "reasoning": "90-day deadline creates urgency. Mid-size lenders need external expertise to avoid penalties and operational disruption."
                    },
                    {
                        "direct_impact": ["Quick Commerce Startups", "Grocery Delivery Apps", "Dark Store Operators", "Last-Mile Logistics"],
                        "direct_impact_reasoning": "Zepto's $200M funding and 50-city expansion directly threatens competitors like Blinkit, Instamart, and BigBasket. They must respond with their own expansion or differentiation strategies.",
                        "indirect_impact": ["Cold Chain Infrastructure", "Warehouse Real Estate", "Gig Economy Platforms", "Packaging Suppliers"],
                        "indirect_impact_reasoning": "More dark stores = more cold storage demand. Warehouse rentals in urban areas will spike. Delivery fleet demand increases (gig platforms). Packaging for quick deliveries needs to scale.",
                        "additional_verticals": ["FMCG Brands", "Local Kirana Tech", "Retail Analytics", "Urban Planning Consultants", "Electric Vehicle Logistics", "Micro-fulfillment Technology"],
                        "additional_verticals_reasoning": "FMCG brands need quick commerce strategy. Kirana stores need tech to compete. Analytics firms help optimize dark store locations. EV logistics for sustainable delivery. Micro-fulfillment tech enables speed.",
                        "positive_sectors": ["Quick Commerce", "Cold Chain", "Logistics Tech", "Warehouse Real Estate"],
                        "negative_sectors": ["Traditional Retail", "Kirana Stores without Tech"],
                        "business_opportunities": [
                            "Competitive intelligence on quick commerce landscape",
                            "Dark store location strategy and feasibility study",
                            "Supply chain optimization for 10-minute delivery",
                            "Market entry strategy for regional quick commerce players",
                            "FMCG brand strategy for quick commerce channel"
                        ],
                        "relevant_services": ["Competitive Intelligence", "Market Intelligence", "Industry Analysis"],
                        "target_roles": ["CEO", "Chief Strategy Officer", "VP Supply Chain", "Director Business Development"],
                        "pitch_angle": "Win the quick commerce battle with strategic intelligence",
                        "reasoning": "Funding war intensifies. Mid-size players need competitive insights to survive or find their niche."
                    },
                    {
                        "direct_impact": ["Semiconductor Manufacturing", "Electronics Assembly", "PCB Manufacturers", "Chip Design Companies"],
                        "direct_impact_reasoning": "Rs 1.26 lakh crore investment directly benefits semiconductor fabs (Tata, Vedanta) and creates demand for local component suppliers. Electronics assemblers can now source chips domestically.",
                        "indirect_impact": ["Electronics Contract Manufacturing", "Consumer Electronics Brands", "Automotive Electronics", "Telecom Equipment"],
                        "indirect_impact_reasoning": "Local chip supply enables contract manufacturers to reduce import dependency. Consumer electronics brands can 'Made in India' premium. Auto electronics benefits from local semiconductor supply. 5G equipment manufacturing becomes viable.",
                        "additional_verticals": ["Specialty Chemicals for Semiconductors", "Cleanroom Equipment", "Industrial Gases", "Water Treatment for Fabs", "Skilled Workforce Training", "Logistics for Sensitive Components"],
                        "additional_verticals_reasoning": "Chip fabs need ultra-pure chemicals, cleanroom infrastructure, specialty gases, treated water systems. Training institutes will prepare workforce. Sensitive component logistics will emerge as a specialty.",
                        "positive_sectors": ["Semiconductors", "Electronics Manufacturing", "Chemical Suppliers", "Industrial Equipment"],
                        "negative_sectors": ["Chip Importers", "Trading Companies"],
                        "business_opportunities": [
                            "Semiconductor ecosystem supplier identification",
                            "Market entry feasibility for specialty chemical companies",
                            "Workforce skill gap analysis for chip manufacturing",
                            "Supply chain mapping for electronics components",
                            "Competitive analysis of global semiconductor players entering India"
                        ],
                        "relevant_services": ["Industry Analysis", "Procurement Intelligence", "Cross Border Expansion"],
                        "target_roles": ["CEO", "VP Manufacturing", "Chief Procurement Officer", "Director Strategy"],
                        "pitch_angle": "Capitalize on India's semiconductor revolution",
                        "reasoning": "Historic opportunity for electronics ecosystem. Companies need market intelligence to position themselves in the emerging value chain."
                    }
                ]
                return json.dumps(mock_impacts[prompt_hash % 3])
            elif "compan" in prompt.lower() or "extract" in prompt.lower() or "find" in prompt.lower() or "real" in prompt.lower():
                # Return multiple MID-SIZE companies (50-300 employees) with INTENT SIGNALS
                prompt_lower = prompt.lower()
                
                # Oil/Energy sector companies WITH INTENT
                if "oil" in prompt_lower or "energy" in prompt_lower or "fuel" in prompt_lower or "petro" in prompt_lower:
                    companies = [
                        {"company_name": "Petrosol Energy Services", "company_size": "mid", "industry": "Oil Equipment", "website": "https://petrosol.in", "description": "Oil field equipment supplier, 180 employees", "intent_signal": "Facing margin pressure, seeking cost optimization", "reason_relevant": "Directly impacted by oil price volatility, needs procurement intelligence"},
                        {"company_name": "Gujarat Oilfield Services", "company_size": "mid", "industry": "Oil Services", "website": "https://gosindia.com", "description": "Oilfield services and equipment, 220 employees", "intent_signal": "Restructuring supply chain operations", "reason_relevant": "Announced restructuring, needs strategic consulting support"},
                        {"company_name": "Deep Industries Ltd", "company_size": "mid", "industry": "Oil & Gas Equipment", "website": "https://deepind.com", "description": "Drilling and production equipment, 190 employees", "intent_signal": "Expanding into new regions", "reason_relevant": "Market expansion plans require competitive intelligence"},
                        {"company_name": "Shree Fuel Agencies", "company_size": "mid", "industry": "Fuel Distribution", "website": "https://shreefuel.in", "description": "Regional fuel distributor, 85 employees", "intent_signal": "Facing pricing pressure from competition", "reason_relevant": "Needs pricing strategy and market analysis"}
                    ]
                # Fintech sector
                elif "fintech" in prompt_lower or "lending" in prompt_lower or "nbfc" in prompt_lower:
                    companies = [
                        {"company_name": "Lendingkart", "company_size": "mid", "industry": "Fintech", "website": "https://lendingkart.com", "description": "SME lending platform with 150+ employees", "reason_relevant": "Directly affected by RBI KYC norms"},
                        {"company_name": "Capital Float", "company_size": "mid", "industry": "Fintech", "website": "https://capitalfloat.com", "description": "Digital lending for businesses, 200 employees", "reason_relevant": "Must comply with new regulations"},
                        {"company_name": "Indifi Technologies", "company_size": "mid", "industry": "Fintech", "website": "https://indifi.com", "description": "Business loans platform, 180 employees", "reason_relevant": "KYC compliance deadline applies"}
                    ]
                elif "quick commerce" in prompt_lower or "logistics" in prompt_lower or "delivery" in prompt_lower:
                    companies = [
                        {"company_name": "Country Delight", "company_size": "mid", "industry": "Quick Commerce", "website": "https://countrydelight.in", "description": "Farm-fresh delivery, 250 employees", "reason_relevant": "Competes in quick delivery space"},
                        {"company_name": "Milkbasket", "company_size": "mid", "industry": "Quick Commerce", "website": "https://milkbasket.com", "description": "Micro-delivery platform, 200 employees", "reason_relevant": "Market pressure from competition"},
                        {"company_name": "Delhivery", "company_size": "mid", "industry": "Logistics", "website": "https://delhivery.com", "description": "Logistics and fulfillment, 300 employees in tech", "reason_relevant": "Last-mile logistics opportunity"}
                    ]
                elif "semiconductor" in prompt_lower or "electronics" in prompt_lower or "manufacturing" in prompt_lower:
                    companies = [
                        {"company_name": "VVDN Technologies", "company_size": "mid", "industry": "Electronics Manufacturing", "website": "https://vvdntech.com", "description": "Electronics design and manufacturing, 250 employees", "reason_relevant": "Benefits from semiconductor ecosystem"},
                        {"company_name": "Syrma SGS Technology", "company_size": "mid", "industry": "Electronics", "website": "https://syrmasgs.com", "description": "EMS provider, 200 employees", "reason_relevant": "Component sourcing opportunity"},
                        {"company_name": "Kaynes Technology", "company_size": "mid", "industry": "Electronics", "website": "https://kaynes.com", "description": "ESDM company, 280 employees", "reason_relevant": "Part of electronics value chain"}
                    ]
                elif "hr" in prompt_lower or "recruitment" in prompt_lower or "talent" in prompt_lower:
                    companies = [
                        {"company_name": "Xpheno", "company_size": "mid", "industry": "HR Tech", "website": "https://xpheno.com", "description": "Specialist staffing, 120 employees", "reason_relevant": "Talent acquisition opportunity"},
                        {"company_name": "Careernet", "company_size": "mid", "industry": "Recruitment", "website": "https://careernet.in", "description": "Recruitment solutions, 180 employees", "reason_relevant": "Hiring market changes"},
                        {"company_name": "PeopleStrong", "company_size": "mid", "industry": "HR Tech", "website": "https://peoplestrong.com", "description": "HR technology platform, 250 employees", "reason_relevant": "Workforce optimization needs"}
                    ]
                elif "regtech" in prompt_lower or "compliance" in prompt_lower:
                    companies = [
                        {"company_name": "Signzy", "company_size": "mid", "industry": "RegTech", "website": "https://signzy.com", "description": "Digital KYC solutions, 150 employees", "reason_relevant": "KYC technology demand surge"},
                        {"company_name": "IDfy", "company_size": "mid", "industry": "RegTech", "website": "https://idfy.com", "description": "Identity verification platform, 180 employees", "reason_relevant": "Compliance solutions provider"},
                        {"company_name": "Perfios", "company_size": "mid", "industry": "RegTech", "website": "https://perfios.com", "description": "Financial data analytics, 200 employees", "reason_relevant": "Regulatory data needs"}
                    ]
                else:
                    # Default mid-size companies across sectors
                    companies = [
                        {"company_name": "Moglix", "company_size": "mid", "industry": "B2B Commerce", "website": "https://moglix.com", "description": "Industrial B2B marketplace, 280 employees", "reason_relevant": "Supply chain intelligence needs"},
                        {"company_name": "OfBusiness", "company_size": "mid", "industry": "B2B Commerce", "website": "https://ofbusiness.com", "description": "B2B raw materials platform, 250 employees", "reason_relevant": "Procurement optimization opportunity"},
                        {"company_name": "Udaan", "company_size": "mid", "industry": "B2B Commerce", "website": "https://udaan.com", "description": "B2B trade platform, 300 employees", "reason_relevant": "Market intelligence needs"}
                    ]
                
                return json.dumps(companies)
            elif "contact" in prompt.lower() or "person" in prompt.lower():
                return json.dumps({
                    "person_name": "Rahul Sharma",
                    "role": "CTO",
                    "linkedin_url": "https://linkedin.com/in/rahul-sharma"
                })
            elif "email" in prompt.lower() or "outreach" in prompt.lower() or "pitch" in prompt.lower():
                mock_emails = [
                    {
                        "subject": "RBI's New KYC Norms - Impact Assessment for Lenders",
                        "body": f"Hi there,\n\nI noticed the RBI's new KYC mandate and thought of your company. With the 90-day compliance deadline, many fintech lenders are scrambling to understand the full impact on their operations.\n\nAt Coherent Market Insights, we've been tracking this regulatory shift closely. We can help with:\n\n• Regulatory compliance landscape assessment\n• Cost-impact analysis of new KYC requirements\n• Benchmarking against industry best practices\n\nWould you be open to a 15-minute call to discuss how we might support your compliance strategy?\n\nBest regards,\nCoherent Market Insights Team"
                    },
                    {
                        "subject": "Quick Commerce Battle - Competitive Intelligence Opportunity",
                        "body": f"Hi there,\n\nWith Zepto's $200M raise and aggressive expansion plans, the quick commerce landscape is shifting rapidly. I thought this might be relevant for your strategic planning.\n\nAt Coherent Market Insights, we help companies navigate competitive disruptions through:\n\n• Competitor profiling and strategy analysis\n• Market share tracking and benchmarking\n• Go-to-market strategy recommendations\n\nWould a 15-minute call be useful to explore how we can help you stay ahead of the competition?\n\nBest regards,\nCoherent Market Insights Team"
                    },
                    {
                        "subject": "Semiconductor Policy - Supply Chain Opportunity Analysis",
                        "body": f"Hi there,\n\nThe government's Rs 1.26 lakh crore semiconductor investment opens significant opportunities for the electronics value chain. I wanted to reach out given your position in the industry.\n\nAt Coherent Market Insights, we specialize in:\n\n• Supply chain opportunity mapping\n• Supplier identification and profiling\n• Market entry feasibility studies\n\nWould you be interested in a 15-minute discussion about how to capitalize on this policy shift?\n\nBest regards,\nCoherent Market Insights Team"
                    }
                ]
                return json.dumps(mock_emails[prompt_hash % 3])
        
        return "Mock LLM response for testing purposes."
    
    def _get_mock_company_response(self, prompt_lower: str) -> str:
        """Return mock company data with intent signals."""
        # Oil/Energy sector companies WITH INTENT
        if "oil" in prompt_lower or "energy" in prompt_lower or "fuel" in prompt_lower or "petro" in prompt_lower:
            companies = [
                {"company_name": "Petrosol Energy Services", "company_size": "mid", "industry": "Oil Equipment", "website": "https://petrosol.in", "description": "Oil field equipment supplier, 180 employees", "intent_signal": "Facing margin pressure, seeking cost optimization", "reason_relevant": "Directly impacted by oil price volatility, needs procurement intelligence"},
                {"company_name": "Gujarat Oilfield Services", "company_size": "mid", "industry": "Oil Services", "website": "https://gosindia.com", "description": "Oilfield services and equipment, 220 employees", "intent_signal": "Restructuring supply chain operations", "reason_relevant": "Announced restructuring, needs strategic consulting support"},
                {"company_name": "Deep Industries Ltd", "company_size": "mid", "industry": "Oil & Gas Equipment", "website": "https://deepind.com", "description": "Drilling and production equipment, 190 employees", "intent_signal": "Expanding into new regions", "reason_relevant": "Market expansion plans require competitive intelligence"},
                {"company_name": "Shree Fuel Agencies", "company_size": "mid", "industry": "Fuel Distribution", "website": "https://shreefuel.in", "description": "Regional fuel distributor, 85 employees", "intent_signal": "Facing pricing pressure from competition", "reason_relevant": "Needs pricing strategy and market analysis"}
            ]
        # Fintech sector
        elif "fintech" in prompt_lower or "lending" in prompt_lower or "nbfc" in prompt_lower:
            companies = [
                {"company_name": "Lendingkart", "company_size": "mid", "industry": "Fintech", "website": "https://lendingkart.com", "description": "SME lending platform, 150 employees", "intent_signal": "Compliance overhaul needed", "reason_relevant": "Directly affected by RBI KYC norms"},
                {"company_name": "Capital Float", "company_size": "mid", "industry": "Fintech", "website": "https://capitalfloat.com", "description": "Digital lending for businesses, 200 employees", "intent_signal": "Seeking regulatory guidance", "reason_relevant": "Must comply with new regulations"},
                {"company_name": "Indifi Technologies", "company_size": "mid", "industry": "Fintech", "website": "https://indifi.com", "description": "Business loans platform, 180 employees", "intent_signal": "Hiring compliance team", "reason_relevant": "KYC compliance deadline applies"}
            ]
        # Quick Commerce/Logistics
        elif "logistics" in prompt_lower or "delivery" in prompt_lower or "commerce" in prompt_lower:
            companies = [
                {"company_name": "Country Delight", "company_size": "mid", "industry": "Quick Commerce", "website": "https://countrydelight.in", "description": "Farm-fresh delivery, 250 employees", "intent_signal": "Expanding to new cities", "reason_relevant": "Competes in quick delivery space"},
                {"company_name": "Delhivery Express", "company_size": "mid", "industry": "Logistics", "website": "https://delhivery.com", "description": "Last-mile logistics, 280 employees", "intent_signal": "Seeking supply chain optimization", "reason_relevant": "Last-mile logistics opportunity"}
            ]
        # Semiconductor/Electronics
        elif "semiconductor" in prompt_lower or "electronics" in prompt_lower or "manufacturing" in prompt_lower:
            companies = [
                {"company_name": "VVDN Technologies", "company_size": "mid", "industry": "Electronics Manufacturing", "website": "https://vvdntech.com", "description": "Electronics design and manufacturing, 250 employees", "intent_signal": "Entering semiconductor supply chain", "reason_relevant": "Benefits from semiconductor ecosystem"},
                {"company_name": "Syrma SGS Technology", "company_size": "mid", "industry": "Electronics", "website": "https://syrmasgs.com", "description": "EMS provider, 200 employees", "intent_signal": "Diversifying component sources", "reason_relevant": "Component sourcing opportunity"},
                {"company_name": "Kaynes Technology", "company_size": "mid", "industry": "Electronics", "website": "https://kaynes.com", "description": "ESDM company, 280 employees", "intent_signal": "Expanding manufacturing capacity", "reason_relevant": "Part of electronics value chain"}
            ]
        # Default - general mid-size companies
        else:
            companies = [
                {"company_name": "Moglix", "company_size": "mid", "industry": "B2B Commerce", "website": "https://moglix.com", "description": "Industrial B2B marketplace, 280 employees", "intent_signal": "Seeking procurement intelligence", "reason_relevant": "Supply chain intelligence needs"},
                {"company_name": "OfBusiness", "company_size": "mid", "industry": "B2B Commerce", "website": "https://ofbusiness.com", "description": "B2B raw materials platform, 250 employees", "intent_signal": "Expanding supplier network", "reason_relevant": "Procurement optimization opportunity"},
                {"company_name": "Udaan", "company_size": "mid", "industry": "B2B Commerce", "website": "https://udaan.com", "description": "B2B trade platform, 300 employees", "intent_signal": "Market expansion underway", "reason_relevant": "Market intelligence needs"}
            ]
        
        return json.dumps(companies)
    
    async def check_ollama_health(self) -> bool:
        """Check if Ollama is available."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self.settings.ollama_base_url}/api/tags")
                return response.status_code == 200
        except Exception:
            return False
    
    async def get_provider_status(self) -> Dict[str, bool]:
        """Get status of all LLM providers."""
        ollama_ok = await self.check_ollama_health() if self.settings.use_ollama else False
        gemini_ok = self._gemini_configured
        
        return {
            "ollama": ollama_ok,
            "gemini": gemini_ok,
            "any_available": ollama_ok or gemini_ok
        }
