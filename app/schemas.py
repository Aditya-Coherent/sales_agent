"""
Pydantic schemas for the India Trend Lead Agent.
Defines data models for trends, companies, contacts, and emails.
"""

from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, Field
from enum import Enum


class Severity(str, Enum):
    """Trend severity levels."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class CompanySize(str, Enum):
    """Company size categories."""
    STARTUP = "startup"
    MID = "mid"
    ENTERPRISE = "enterprise"


class TrendData(BaseModel):
    """Market trend detected from RSS/Tavily."""
    id: str = Field(default="")
    trend_title: str
    summary: str
    severity: Severity = Severity.MEDIUM
    industries_affected: List[str] = Field(default_factory=list)
    source_links: List[str] = Field(default_factory=list)
    keywords: List[str] = Field(default_factory=list)
    detected_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        use_enum_values = True


class ImpactAnalysis(BaseModel):
    """Deep mid-size company focused impact analysis."""
    trend_id: str
    trend_title: str
    
    # Part 1: Direct Impact on Mid-Size Companies
    direct_impact: List[str] = Field(default_factory=list)  # Specific mid-size company types
    direct_impact_reasoning: str = ""  # Their specific challenges
    
    # Part 2: Indirect Impact (second-order effects on mid-size companies)
    indirect_impact: List[str] = Field(default_factory=list)
    indirect_impact_reasoning: str = ""
    
    # Part 3: Additional Industry Verticals (non-obvious mid-size company types)
    additional_verticals: List[str] = Field(default_factory=list)
    additional_verticals_reasoning: str = ""
    
    # NEW: Mid-size company specific pain points
    midsize_pain_points: List[str] = Field(default_factory=list)
    
    # NEW: Specific consulting project deliverables
    consulting_projects: List[str] = Field(default_factory=list)
    
    # Consulting opportunities
    positive_sectors: List[str] = Field(default_factory=list)
    negative_sectors: List[str] = Field(default_factory=list)
    business_opportunities: List[str] = Field(default_factory=list)  # Legacy, kept for compatibility
    relevant_services: List[str] = Field(default_factory=list)
    target_roles: List[str] = Field(default_factory=list)
    pitch_angle: str = ""
    reasoning: str = ""


class CompanyData(BaseModel):
    """Company information found via search."""
    id: str = Field(default="")
    company_name: str
    company_size: CompanySize = CompanySize.MID
    industry: str
    website: str = ""
    domain: str = ""
    description: str = ""
    reason_relevant: str = ""
    trend_id: str = ""
    
    class Config:
        use_enum_values = True


class ContactData(BaseModel):
    """Decision-maker contact information."""
    id: str = Field(default="")
    company_id: str = ""
    company_name: str
    person_name: str
    role: str
    linkedin_url: str = ""
    email: str = ""
    email_confidence: int = 0
    email_source: str = ""  # "apollo", "hunter", "pattern"
    verified: bool = False


class OutreachEmail(BaseModel):
    """Generated personalized email."""
    id: str = Field(default="")
    contact_id: str = ""
    trend_title: str
    company_name: str
    person_name: str
    role: str
    email: str
    subject: str
    body: str
    email_confidence: int = 0
    generated_at: datetime = Field(default_factory=datetime.utcnow)


class LeadRecord(BaseModel):
    """Complete lead record combining all data."""
    trend: TrendData
    impact: ImpactAnalysis
    company: CompanyData
    contact: ContactData
    outreach: OutreachEmail


class PipelineResult(BaseModel):
    """Result of running the full pipeline."""
    status: str = "success"
    leads_generated: int = 0
    trends_detected: int = 0
    companies_found: int = 0
    emails_found: int = 0
    output_file: str = ""
    errors: List[str] = Field(default_factory=list)
    run_time_seconds: float = 0.0
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class EmailFinderResult(BaseModel):
    """Result from email finding APIs."""
    email: str = ""
    confidence: int = 0
    source: str = ""  # "apollo", "hunter", "pattern"
    verified: bool = False
    error: str = ""


class AgentState(BaseModel):
    """State passed between agents in the pipeline."""
    trends: List[TrendData] = Field(default_factory=list)
    impacts: List[ImpactAnalysis] = Field(default_factory=list)
    companies: List[CompanyData] = Field(default_factory=list)
    contacts: List[ContactData] = Field(default_factory=list)
    outreach_emails: List[OutreachEmail] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    current_step: str = "init"
