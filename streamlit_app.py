"""
India Trend Lead Generation Agent - Streamlit Interactive Dashboard
Human-in-the-loop workflow for testing and controlling the agent pipeline.
"""

import streamlit as st
import asyncio
import json
import pandas as pd
from datetime import datetime
from pathlib import Path
import sys

# Add app to path
sys.path.insert(0, str(Path(__file__).parent))

from app.config import get_settings
from app.schemas import AgentState, TrendData, ImpactAnalysis, CompanyData, ContactData, OutreachEmail
from app.tools.llm_tool import LLMTool
from app.tools.rss_tool import RSSTool
from app.tools.tavily_tool import TavilyTool
from app.agents.trend_agent import TrendAgent
from app.agents.impact_agent import ImpactAgent
from app.agents.company_agent import CompanyAgent
from app.agents.contact_agent import ContactAgent
from app.agents.email_agent import EmailAgent

# Page config
st.set_page_config(
    page_title="🇮🇳 India Lead Gen Agent",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    /* Main theme */
    .stApp {
        background: linear-gradient(135deg, #0f0f23 0%, #1a1a3e 100%);
    }
    
    /* Cards */
    .trend-card {
        background: linear-gradient(145deg, #1e1e3f, #2a2a5a);
        border-radius: 12px;
        padding: 20px;
        margin: 10px 0;
        border-left: 4px solid #00d4ff;
        box-shadow: 0 4px 15px rgba(0, 212, 255, 0.1);
    }
    
    .company-card {
        background: linear-gradient(145deg, #1e3f1e, #2a5a2a);
        border-radius: 12px;
        padding: 20px;
        margin: 10px 0;
        border-left: 4px solid #00ff88;
        box-shadow: 0 4px 15px rgba(0, 255, 136, 0.1);
    }
    
    .contact-card {
        background: linear-gradient(145deg, #3f1e3f, #5a2a5a);
        border-radius: 12px;
        padding: 20px;
        margin: 10px 0;
        border-left: 4px solid #ff00ff;
        box-shadow: 0 4px 15px rgba(255, 0, 255, 0.1);
    }
    
    .email-card {
        background: linear-gradient(145deg, #3f3f1e, #5a5a2a);
        border-radius: 12px;
        padding: 20px;
        margin: 10px 0;
        border-left: 4px solid #ffcc00;
        box-shadow: 0 4px 15px rgba(255, 204, 0, 0.1);
    }
    
    /* Severity badges */
    .badge-high {
        background: #ff4757;
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: bold;
    }
    
    .badge-medium {
        background: #ffa502;
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: bold;
    }
    
    .badge-low {
        background: #2ed573;
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: bold;
    }
    
    /* Progress steps */
    .step-active {
        color: #00d4ff;
        font-weight: bold;
    }
    
    .step-completed {
        color: #00ff88;
    }
    
    .step-pending {
        color: #666;
    }
    
    /* Stats */
    .stat-box {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 10px;
        padding: 15px;
        text-align: center;
    }
    
    .stat-number {
        font-size: 2.5em;
        font-weight: bold;
        color: #00d4ff;
    }
    
    .stat-label {
        color: #888;
        font-size: 0.9em;
    }
    
    /* Hide streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


def init_session_state():
    """Initialize session state variables."""
    defaults = {
        'current_step': 0,
        'mock_mode': True,
        'trends': [],
        'selected_trends': [],
        'impacts': [],
        'companies': [],
        'selected_companies': [],
        'contacts': [],
        'selected_contacts': [],
        'outreach_emails': [],
        'logs': [],
        'pipeline_running': False,
        'agent_state': None
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def add_log(message: str, level: str = "info"):
    """Add a log message."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    icons = {"info": "ℹ️", "success": "✅", "warning": "⚠️", "error": "❌"}
    st.session_state.logs.append({
        "time": timestamp,
        "level": level,
        "icon": icons.get(level, "📝"),
        "message": message
    })


def render_sidebar():
    """Render the sidebar with controls and status."""
    with st.sidebar:
        st.markdown("# 🎯 CMI Sales Agent")
        st.markdown("*Coherent Market Insights*")
        st.markdown("---")
        
        # Mode toggle
        st.markdown("### ⚙️ Settings")
        st.session_state.mock_mode = st.toggle(
            "Mock Mode",
            value=st.session_state.mock_mode,
            help="Use mock data instead of real API calls"
        )
        
        if st.session_state.mock_mode:
            st.info("🔧 Mock mode: No API credits used")
        else:
            st.warning("⚡ Live mode: Real API calls")
        
        # Pipeline steps - Consultant Flow
        st.markdown("---")
        st.markdown("### 📋 Consultant Pipeline")
        
        steps = [
            ("1️⃣", "News Detection", 0),
            ("2️⃣", "Opportunity Analysis", 1),
            ("3️⃣", "Target Companies", 2),
            ("4️⃣", "Decision Makers", 3),
            ("5️⃣", "Pitch Generation", 4)
        ]
        
        for icon, name, idx in steps:
            if idx < st.session_state.current_step:
                st.markdown(f"<span class='step-completed'>✓ {icon} {name}</span>", unsafe_allow_html=True)
            elif idx == st.session_state.current_step:
                st.markdown(f"<span class='step-active'>→ {icon} {name}</span>", unsafe_allow_html=True)
            else:
                st.markdown(f"<span class='step-pending'>○ {icon} {name}</span>", unsafe_allow_html=True)
        
        # Stats
        st.markdown("---")
        st.markdown("### 📊 Current Stats")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Trends", len(st.session_state.trends))
            st.metric("Companies", len(st.session_state.companies))
        with col2:
            st.metric("Contacts", len(st.session_state.contacts))
            st.metric("Emails", len(st.session_state.outreach_emails))
        
        # Reset button
        st.markdown("---")
        if st.button("🔄 Reset Pipeline", type="secondary", use_container_width=True):
            for key in ['current_step', 'trends', 'selected_trends', 'impacts', 
                       'companies', 'selected_companies', 'contacts', 
                       'selected_contacts', 'outreach_emails', 'logs', 'agent_state']:
                if key == 'current_step':
                    st.session_state[key] = 0
                elif key == 'mock_mode':
                    continue
                else:
                    st.session_state[key] = []
            st.rerun()


def render_step_0_trends():
    """Step 0: Trend Detection."""
    today = datetime.now().strftime("%A, %B %d, %Y")
    st.markdown("## 📰 Step 1: Today's News Detection")
    st.markdown(f"**📅 {today}** - Fetching breaking Indian business news from the last 24 hours.")
    
    col1, col2 = st.columns([3, 1])
    
    with col2:
        if st.button("🚀 Detect Trends", type="primary", use_container_width=True):
            with st.spinner("Detecting trends..."):
                add_log("Starting trend detection...", "info")
                
                # Run trend agent
                async def run():
                    agent = TrendAgent(mock_mode=st.session_state.mock_mode)
                    state = AgentState()
                    result = await agent.detect_trends(state)
                    return result.trends
                
                trends = asyncio.run(run())
                st.session_state.trends = trends
                add_log(f"Detected {len(trends)} trends", "success")
                st.rerun()
    
    if st.session_state.trends:
        st.markdown("### 📰 Detected Trends")
        st.markdown("**Select the trends you want to pursue:**")
        
        selected = []
        for i, trend in enumerate(st.session_state.trends):
            with st.container():
                col1, col2 = st.columns([0.1, 0.9])
                
                with col1:
                    checked = st.checkbox(
                        "Select",
                        key=f"trend_{i}",
                        value=True,
                        label_visibility="collapsed"
                    )
                    if checked:
                        selected.append(trend)
                
                with col2:
                    severity_class = f"badge-{trend.severity.value if hasattr(trend.severity, 'value') else trend.severity}"
                    severity_val = trend.severity.value.upper() if hasattr(trend.severity, 'value') else str(trend.severity).upper()
                    
                    # Get source links if available
                    sources = trend.source_links[:2] if trend.source_links else []
                    source_html = ""
                    if sources:
                        source_html = f'<div style="margin-top: 10px; font-size: 12px; color: #888;">📰 Sources: {len(trend.source_links)} articles</div>'
                    
                    # Get keywords
                    keywords = trend.keywords[:5] if trend.keywords else []
                    keywords_html = ""
                    if keywords:
                        keywords_html = '<div style="margin-top: 8px;">' + ''.join([f'<span style="background: rgba(255,255,255,0.1); padding: 2px 8px; border-radius: 10px; font-size: 11px; margin-right: 5px;">#{kw}</span>' for kw in keywords]) + '</div>'
                    
                    st.markdown(f"""
                    <div class="trend-card">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <h4 style="margin: 0; color: #00d4ff;">{trend.trend_title}</h4>
                            <span class="{severity_class}">{severity_val}</span>
                        </div>
                        <p style="color: #ccc; margin: 10px 0;">{trend.summary}</p>
                        <div style="display: flex; gap: 10px; flex-wrap: wrap;">
                            {''.join([f'<span style="background: rgba(0,212,255,0.2); padding: 4px 10px; border-radius: 15px; font-size: 12px;">{ind}</span>' for ind in trend.industries_affected[:4]])}
                        </div>
                        {keywords_html}
                        {source_html}
                    </div>
                    """, unsafe_allow_html=True)
        
        st.session_state.selected_trends = selected
        
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button("✅ Continue with Selected Trends", type="primary", use_container_width=True):
                if st.session_state.selected_trends:
                    st.session_state.current_step = 1
                    add_log(f"Selected {len(st.session_state.selected_trends)} trends to pursue", "info")
                    st.rerun()
                else:
                    st.error("Please select at least one trend")


def render_step_1_impacts():
    """Step 1: Impact Analysis."""
    st.markdown("## 📊 Step 2: Impact Analysis")
    st.markdown("Analyze sector impact and identify business opportunities.")
    
    if not st.session_state.impacts:
        col1, col2 = st.columns([3, 1])
        with col2:
            if st.button("🔬 Analyze Impacts", type="primary", use_container_width=True):
                with st.spinner("Analyzing sector impacts..."):
                    add_log("Starting impact analysis...", "info")
                    
                    async def run():
                        agent = ImpactAgent(mock_mode=st.session_state.mock_mode)
                        state = AgentState(trends=st.session_state.selected_trends)
                        result = await agent.analyze_impacts(state)
                        return result.impacts
                    
                    impacts = asyncio.run(run())
                    st.session_state.impacts = impacts
                    add_log(f"Analyzed {len(impacts)} trend impacts", "success")
                    st.rerun()
    
    if st.session_state.impacts:
        st.markdown("### 🏢 Mid-Size Company Impact Analysis")
        st.markdown("*Focused on companies with 50-300 employees - CMI's sweet spot*")
        
        for impact in st.session_state.impacts:
            with st.expander(f"📌 {impact.trend_title}", expanded=True):
                
                # Part 1: Direct Impact on Mid-Size Companies
                st.markdown("#### 🎯 PART 1: Mid-Size Companies Directly Affected")
                st.markdown("*Specific company types that will feel immediate impact*")
                if hasattr(impact, 'direct_impact') and impact.direct_impact:
                    for company_type in impact.direct_impact:
                        st.markdown(f"• **{company_type}**")
                    if hasattr(impact, 'direct_impact_reasoning') and impact.direct_impact_reasoning:
                        st.info(f"💡 **Their Challenges:** {impact.direct_impact_reasoning}")
                else:
                    for sector in impact.positive_sectors[:4]:
                        st.markdown(f"• **{sector}**")
                
                st.markdown("---")
                
                # Part 2: Indirect Impact
                st.markdown("#### 🔄 PART 2: Second-Order Effects on Mid-Size Companies")
                st.markdown("*Companies affected as a consequence of direct impacts*")
                if hasattr(impact, 'indirect_impact') and impact.indirect_impact:
                    for company_type in impact.indirect_impact:
                        st.markdown(f"• **{company_type}**")
                    if hasattr(impact, 'indirect_impact_reasoning') and impact.indirect_impact_reasoning:
                        st.warning(f"🔗 **Chain Effect:** {impact.indirect_impact_reasoning}")
                
                st.markdown("---")
                
                # Part 3: Non-Obvious Verticals
                st.markdown("#### 🌐 PART 3: Non-Obvious Mid-Size Company Opportunities")
                st.markdown("*Companies most analysts would miss*")
                if hasattr(impact, 'additional_verticals') and impact.additional_verticals:
                    cols = st.columns(2)
                    for i, company_type in enumerate(impact.additional_verticals):
                        with cols[i % 2]:
                            st.markdown(f"• {company_type}")
                    if hasattr(impact, 'additional_verticals_reasoning') and impact.additional_verticals_reasoning:
                        st.success(f"🔍 **Hidden Connections:** {impact.additional_verticals_reasoning}")
                
                st.markdown("---")
                
                # NEW: Mid-Size Pain Points
                if hasattr(impact, 'midsize_pain_points') and impact.midsize_pain_points:
                    st.markdown("#### 😰 Mid-Size Company Pain Points")
                    st.markdown("*Specific challenges where CMI can help*")
                    for pain in impact.midsize_pain_points:
                        st.error(f"🔴 {pain}")
                    st.markdown("---")
                
                # Consulting Projects
                st.markdown("#### 💼 Specific Consulting Deliverables")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**📊 Projects We Can Pitch:**")
                    projects = impact.consulting_projects if hasattr(impact, 'consulting_projects') and impact.consulting_projects else impact.business_opportunities
                    for proj in projects[:5]:
                        st.markdown(f"• {proj}")
                
                with col2:
                    st.markdown("**🎯 Target Decision Makers:**")
                    for role in impact.target_roles:
                        st.markdown(f"• {role}")
                    
                    if hasattr(impact, 'relevant_services') and impact.relevant_services:
                        st.markdown("**📦 CMI Services:**")
                        for svc in impact.relevant_services:
                            st.markdown(f"• {svc}")
                
                # Pitch angle and reasoning
                if hasattr(impact, 'pitch_angle') and impact.pitch_angle:
                    st.markdown(f"**🚀 Pitch Angle:** _{impact.pitch_angle}_")
                
                if hasattr(impact, 'reasoning') and impact.reasoning:
                    st.markdown(f"**💡 Why Act Now:** {impact.reasoning}")
        
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            if st.button("⬅️ Back to Trends", use_container_width=True):
                st.session_state.current_step = 0
                st.rerun()
        with col3:
            if st.button("Continue to Companies ➡️", type="primary", use_container_width=True):
                st.session_state.current_step = 2
                add_log("Moving to company discovery", "info")
                st.rerun()


def render_step_2_companies():
    """Step 2: Company Discovery."""
    st.markdown("## 🏢 Step 3: Company Discovery")
    st.markdown("Find relevant Indian companies for each impacted sector.")
    
    if not st.session_state.companies:
        col1, col2 = st.columns([3, 1])
        with col2:
            if st.button("🔎 Find Companies", type="primary", use_container_width=True):
                with st.spinner("Discovering companies..."):
                    add_log("Starting company discovery...", "info")
                    
                    async def run():
                        agent = CompanyAgent(mock_mode=st.session_state.mock_mode)
                        state = AgentState(
                            trends=st.session_state.selected_trends,
                            impacts=st.session_state.impacts
                        )
                        result = await agent.find_companies(state)
                        return result.companies
                    
                    companies = asyncio.run(run())
                    st.session_state.companies = companies
                    add_log(f"Discovered {len(companies)} companies", "success")
                    st.rerun()
    
    if st.session_state.companies:
        st.markdown("### 🏭 Discovered Companies")
        st.markdown("**Select companies to find contacts for:**")
        
        selected = []
        
        # Group by industry
        industries = {}
        for company in st.session_state.companies:
            ind = company.industry or "Other"
            if ind not in industries:
                industries[ind] = []
            industries[ind].append(company)
        
        for industry, companies in industries.items():
            st.markdown(f"#### 📁 {industry}")
            
            for i, company in enumerate(companies):
                col1, col2 = st.columns([0.1, 0.9])
                
                with col1:
                    checked = st.checkbox(
                        "Select",
                        key=f"company_{company.id}",
                        value=True,
                        label_visibility="collapsed"
                    )
                    if checked:
                        selected.append(company)
                
                with col2:
                    size_emoji = {"startup": "🚀", "mid": "📊", "enterprise": "🏛️"}
                    size_val = company.company_size.value if hasattr(company.company_size, 'value') else company.company_size
                    
                    st.markdown(f"""
                    <div class="company-card">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <h4 style="margin: 0; color: #00ff88;">{company.company_name}</h4>
                            <span>{size_emoji.get(size_val, '📊')} {size_val.title()}</span>
                        </div>
                        <p style="color: #aaa; margin: 5px 0;">{company.description or company.reason_relevant}</p>
                        <div style="display: flex; gap: 20px; font-size: 13px; color: #888;">
                            <span>🌐 {company.domain or 'No domain'}</span>
                            <span>🔗 <a href="{company.website}" target="_blank" style="color: #00ff88;">Website</a></span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
        
        st.session_state.selected_companies = selected
        
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            if st.button("⬅️ Back to Impacts", use_container_width=True):
                st.session_state.current_step = 1
                st.rerun()
        with col3:
            if st.button(f"Find Contacts ({len(selected)} companies) ➡️", type="primary", use_container_width=True):
                if selected:
                    st.session_state.current_step = 3
                    add_log(f"Selected {len(selected)} companies for contact finding", "info")
                    st.rerun()
                else:
                    st.error("Please select at least one company")


def render_step_3_contacts():
    """Step 3: Contact Finding."""
    st.markdown("## 👤 Step 4: Contact Finding")
    st.markdown("Find decision-makers at selected companies.")
    
    if not st.session_state.contacts:
        col1, col2 = st.columns([3, 1])
        with col2:
            if st.button("🔍 Find Contacts", type="primary", use_container_width=True):
                with st.spinner("Finding decision-makers..."):
                    add_log("Starting contact search...", "info")
                    
                    async def run():
                        agent = ContactAgent(mock_mode=st.session_state.mock_mode)
                        # Convert objects to dicts for AgentState
                        trends_data = [t.model_dump() if hasattr(t, 'model_dump') else t for t in st.session_state.selected_trends]
                        impacts_data = [i.model_dump() if hasattr(i, 'model_dump') else i for i in st.session_state.impacts]
                        companies_data = [c.model_dump() if hasattr(c, 'model_dump') else c for c in st.session_state.selected_companies]
                        
                        # Reconstruct proper objects
                        trends = [TrendData(**t) if isinstance(t, dict) else t for t in trends_data]
                        impacts = [ImpactAnalysis(**i) if isinstance(i, dict) else i for i in impacts_data]
                        companies = [CompanyData(**c) if isinstance(c, dict) else c for c in companies_data]
                        
                        state = AgentState(
                            trends=trends,
                            impacts=impacts,
                            companies=companies
                        )
                        result = await agent.find_contacts(state)
                        return result.contacts
                    
                    contacts = asyncio.run(run())
                    st.session_state.contacts = contacts
                    add_log(f"Found {len(contacts)} contacts", "success")
                    st.rerun()
    
    if st.session_state.contacts:
        st.markdown("### 👥 Found Contacts")
        st.markdown("**Select contacts for email outreach:**")
        
        selected = []
        
        for contact in st.session_state.contacts:
            col1, col2 = st.columns([0.1, 0.9])
            
            with col1:
                checked = st.checkbox(
                    "Select",
                    key=f"contact_{contact.id}",
                    value=True,
                    label_visibility="collapsed"
                )
                if checked:
                    selected.append(contact)
            
            with col2:
                email_status = "✅" if contact.email else "⏳ Pending"
                linkedin_link = f'<a href="{contact.linkedin_url}" target="_blank" style="color: #0077b5;">LinkedIn</a>' if contact.linkedin_url else "N/A"
                
                st.markdown(f"""
                <div class="contact-card">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <h4 style="margin: 0; color: #ff00ff;">{contact.person_name}</h4>
                        <span style="font-size: 12px;">Email: {email_status}</span>
                    </div>
                    <p style="color: #ccc; margin: 5px 0;">
                        <strong>{contact.role}</strong> at <strong>{contact.company_name}</strong>
                    </p>
                    <div style="display: flex; gap: 20px; font-size: 13px; color: #888;">
                        <span>💼 {linkedin_link}</span>
                        {f'<span>📧 {contact.email}</span>' if contact.email else ''}
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        st.session_state.selected_contacts = selected
        
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            if st.button("⬅️ Back to Companies", use_container_width=True):
                st.session_state.current_step = 2
                st.rerun()
        with col3:
            if st.button(f"Generate Emails ({len(selected)} contacts) ➡️", type="primary", use_container_width=True):
                if selected:
                    st.session_state.current_step = 4
                    add_log(f"Selected {len(selected)} contacts for email generation", "info")
                    st.rerun()
                else:
                    st.error("Please select at least one contact")


def render_step_4_emails():
    """Step 4: Email Generation."""
    st.markdown("## ✉️ Step 5: Email Generation")
    st.markdown("Find verified emails and generate personalized outreach.")
    
    if not st.session_state.outreach_emails:
        col1, col2 = st.columns([3, 1])
        with col2:
            if st.button("📧 Generate Emails", type="primary", use_container_width=True):
                with st.spinner("Finding emails and generating outreach..."):
                    add_log("Starting email generation...", "info")
                    
                    async def run():
                        agent = EmailAgent(mock_mode=st.session_state.mock_mode)
                        
                        # Convert objects to dicts and back for proper validation
                        trends_data = [t.model_dump() if hasattr(t, 'model_dump') else t for t in st.session_state.selected_trends]
                        impacts_data = [i.model_dump() if hasattr(i, 'model_dump') else i for i in st.session_state.impacts]
                        companies_data = [c.model_dump() if hasattr(c, 'model_dump') else c for c in st.session_state.selected_companies]
                        contacts_data = [c.model_dump() if hasattr(c, 'model_dump') else c for c in st.session_state.selected_contacts]
                        
                        trends = [TrendData(**t) if isinstance(t, dict) else t for t in trends_data]
                        impacts = [ImpactAnalysis(**i) if isinstance(i, dict) else i for i in impacts_data]
                        companies = [CompanyData(**c) if isinstance(c, dict) else c for c in companies_data]
                        contacts = [ContactData(**c) if isinstance(c, dict) else c for c in contacts_data]
                        
                        state = AgentState(
                            trends=trends,
                            impacts=impacts,
                            companies=companies,
                            contacts=contacts
                        )
                        result = await agent.process_emails(state)
                        return result.outreach_emails, result.contacts
                    
                    emails, updated_contacts = asyncio.run(run())
                    st.session_state.outreach_emails = emails
                    st.session_state.contacts = updated_contacts
                    add_log(f"Generated {len(emails)} outreach emails", "success")
                    st.rerun()
    
    if st.session_state.outreach_emails:
        st.markdown("### 📬 Generated Outreach Emails")
        
        for email in st.session_state.outreach_emails:
            confidence_color = "#00ff88" if email.email_confidence >= 70 else "#ffa502" if email.email_confidence >= 50 else "#ff4757"
            
            with st.expander(f"📧 {email.person_name} @ {email.company_name}", expanded=True):
                col1, col2, col3 = st.columns([2, 1, 1])
                with col1:
                    st.markdown(f"**To:** {email.email}")
                with col2:
                    st.markdown(f"**Role:** {email.role}")
                with col3:
                    st.markdown(f"**Confidence:** <span style='color: {confidence_color}'>{email.email_confidence}%</span>", unsafe_allow_html=True)
                
                st.markdown("---")
                st.markdown(f"**Subject:** `{email.subject}`")
                st.markdown("**Body:**")
                st.code(email.body, language=None)
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button(f"📋 Copy Email", key=f"copy_{email.id}"):
                        st.toast("Email copied! (Simulated)")
                with col2:
                    if st.button(f"✏️ Edit", key=f"edit_{email.id}"):
                        st.info("Edit functionality coming soon!")
        
        st.markdown("---")
        
        # Export options
        st.markdown("### 📥 Export Results")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # JSON export
            export_data = {
                "generated_at": datetime.now().isoformat(),
                "trends": [t.model_dump() for t in st.session_state.selected_trends],
                "companies": [c.model_dump() for c in st.session_state.selected_companies],
                "contacts": [c.model_dump() for c in st.session_state.contacts],
                "outreach_emails": [e.model_dump() for e in st.session_state.outreach_emails]
            }
            st.download_button(
                label="📄 Download JSON",
                data=json.dumps(export_data, indent=2, default=str),
                file_name=f"leads_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True
            )
        
        with col2:
            # CSV export
            csv_data = []
            for email in st.session_state.outreach_emails:
                csv_data.append({
                    "Company": email.company_name,
                    "Contact": email.person_name,
                    "Role": email.role,
                    "Email": email.email,
                    "Confidence": email.email_confidence,
                    "Subject": email.subject,
                    "Body": email.body[:200] + "..."
                })
            df = pd.DataFrame(csv_data)
            st.download_button(
                label="📊 Download CSV",
                data=df.to_csv(index=False),
                file_name=f"leads_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col3:
            if st.button("🔄 Start New Run", type="primary", use_container_width=True):
                for key in ['current_step', 'trends', 'selected_trends', 'impacts', 
                           'companies', 'selected_companies', 'contacts', 
                           'selected_contacts', 'outreach_emails']:
                    if key == 'current_step':
                        st.session_state[key] = 0
                    else:
                        st.session_state[key] = []
                st.rerun()


def render_logs():
    """Render the activity log."""
    with st.expander("📜 Activity Log", expanded=False):
        if st.session_state.logs:
            for log in reversed(st.session_state.logs[-20:]):
                st.markdown(f"`{log['time']}` {log['icon']} {log['message']}")
        else:
            st.info("No activity yet. Start the pipeline to see logs.")


def main():
    """Main application."""
    init_session_state()
    render_sidebar()
    
    # Header - Coherent Market Insights Branding
    today = datetime.now().strftime("%A, %B %d, %Y")
    st.markdown(f"""
    <div style="text-align: center; padding: 20px 0;">
        <h1 style="background: linear-gradient(90deg, #0066cc, #00aaff, #00d4ff); 
                   -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                   font-size: 2.2em; margin-bottom: 5px;">
            Coherent Market Insights
        </h1>
        <h2 style="color: #00d4ff; font-size: 1.4em; margin-top: 0;">
            🎯 Consulting Lead Generation Agent
        </h2>
        <p style="color: #888; font-size: 1.1em;">
            📅 <strong>{today}</strong> | News → Opportunity Analysis → Target Companies → Personalized Pitch
        </p>
        <p style="color: #666; font-size: 0.9em; margin-top: 5px;">
            Market Intelligence • Competitive Analysis • Strategic Advisory
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Progress bar
    progress = st.session_state.current_step / 4
    st.progress(progress, text=f"Pipeline Progress: {int(progress * 100)}%")
    
    st.markdown("---")
    
    # Render current step
    if st.session_state.current_step == 0:
        render_step_0_trends()
    elif st.session_state.current_step == 1:
        render_step_1_impacts()
    elif st.session_state.current_step == 2:
        render_step_2_companies()
    elif st.session_state.current_step == 3:
        render_step_3_contacts()
    elif st.session_state.current_step == 4:
        render_step_4_emails()
    
    # Logs at bottom
    st.markdown("---")
    render_logs()


if __name__ == "__main__":
    main()
