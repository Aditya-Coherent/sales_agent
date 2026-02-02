# 🇮🇳 India Trend Lead Generation Agent

AI-powered market trend detection and B2B lead generation system for Indian companies.

## 🎯 What It Does

This system automatically:
1. **Detects Market Trends** - Monitors Google News RSS and validates with Tavily
2. **Analyzes Impact** - Identifies which sectors win/lose from each trend
3. **Finds Companies** - Discovers relevant Indian companies by sector
4. **Locates Decision Makers** - Finds CTOs, CEOs, and other key contacts
5. **Finds Verified Emails** - Uses Apollo.io (primary) and Hunter.io (fallback)
6. **Generates Outreach** - Creates personalized emails for each lead

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    LLM PROVIDER (Smart Fallback)                │
│  Ollama (Mistral) → Gemini API                                 │
└─────────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────────┐
│                   LANGGRAPH PIPELINE                            │
│                                                                 │
│  Trend Agent → Impact Agent → Company Agent → Contact Agent    │
│                                                   ↓             │
│                                            Email Agent          │
│                                                   ↓             │
│                                         JSON + CSV Output       │
└─────────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
cd india-trend-lead-agent
pip install -r requirements.txt
```

### 🖥️ Interactive Streamlit Dashboard (Recommended)

The easiest way to use the agent is through the interactive Streamlit dashboard:

```bash
streamlit run streamlit_app.py
```

This gives you:
- **Step-by-step execution** with visual feedback
- **Manual selection** of trends, companies, and contacts at each step
- **Real-time logs** and progress tracking
- **Export to JSON/CSV** with one click
- **Mock mode toggle** for testing without API calls

### 2. Configure Environment

Create a `.env` file in the project root:

```env
# LLM Configuration
USE_OLLAMA=true
OLLAMA_MODEL=mistral
OLLAMA_BASE_URL=http://localhost:11434
GEMINI_API_KEY=your_gemini_key

# Search API
TAVILY_API_KEY=your_tavily_key

# Email Finder APIs
APOLLO_API_KEY=your_apollo_key
HUNTER_API_KEY=your_hunter_key

# Settings
COUNTRY=India
MAX_TRENDS=3
MAX_COMPANIES_PER_TREND=3
MAX_CONTACTS_PER_COMPANY=2
EMAIL_CONFIDENCE_THRESHOLD=70
MOCK_MODE=false
DATABASE_URL=sqlite+aiosqlite:///./leads.db
```

### 3. Run the Pipeline

**Option A: Command Line**
```bash
# Real mode (uses APIs)
python -m app.main

# Mock mode (no API calls, for testing)
python -m app.main --mock
```

**Option B: FastAPI Server**
```bash
# Start server
python -m app.main --server --port 8000

# Then call the API
curl -X POST http://localhost:8000/run
```

## 📡 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Health check |
| `/health` | GET | Detailed health with LLM status |
| `/run` | POST | Execute the full pipeline |
| `/results` | GET | Get latest pipeline results |
| `/leads` | GET | Get leads from database |
| `/outputs` | GET | List output files |
| `/outputs/{file}` | GET | Download output file |

### Run Pipeline via API

```bash
# Run with default settings
curl -X POST http://localhost:8000/run

# Run in mock mode
curl -X POST http://localhost:8000/run \
  -H "Content-Type: application/json" \
  -d '{"mock_mode": true}'
```

## 📁 Project Structure

```
india-trend-lead-agent/
├── app/
│   ├── main.py              # FastAPI + CLI entry point
│   ├── config.py            # Environment configuration
│   ├── schemas.py           # Pydantic data models
│   ├── database.py          # SQLite operations
│   │
│   ├── agents/
│   │   ├── trend_agent.py   # RSS + Tavily trend detection
│   │   ├── impact_agent.py  # Sector impact analysis
│   │   ├── company_agent.py # Company discovery
│   │   ├── contact_agent.py # Decision-maker finding
│   │   ├── email_agent.py   # Email finder + writer
│   │   └── orchestrator.py  # LangGraph pipeline
│   │
│   ├── tools/
│   │   ├── llm_tool.py      # Ollama + Gemini wrapper
│   │   ├── rss_tool.py      # Google News RSS
│   │   ├── tavily_tool.py   # Tavily search
│   │   ├── apollo_tool.py   # Apollo.io email finder
│   │   ├── hunter_tool.py   # Hunter.io backup
│   │   └── domain_utils.py  # Domain extraction
│   │
│   └── outputs/             # Generated leads
│
├── requirements.txt
├── .env                     # Your API keys
└── README.md
```

## 🔧 Configuration Options

| Variable | Default | Description |
|----------|---------|-------------|
| `USE_OLLAMA` | `true` | Use local Ollama model |
| `OLLAMA_MODEL` | `mistral` | Ollama model name |
| `MAX_TRENDS` | `3` | Max trends to process |
| `MAX_COMPANIES_PER_TREND` | `3` | Companies per trend |
| `MAX_CONTACTS_PER_COMPANY` | `2` | Contacts per company |
| `EMAIL_CONFIDENCE_THRESHOLD` | `70` | Min email confidence |
| `MOCK_MODE` | `false` | Use mock data |

## 📊 Output Format

### JSON Output (leads_YYYYMMDD_HHMMSS.json)

```json
{
  "run_id": "20260127_143022",
  "total_leads": 15,
  "leads": [
    {
      "trend": {
        "title": "AI Adoption Surge in Indian Enterprises",
        "summary": "...",
        "industries": ["Technology", "BFSI"]
      },
      "company": {
        "name": "Zoho Corporation",
        "industry": "Technology",
        "website": "https://zoho.com",
        "domain": "zoho.com"
      },
      "contact": {
        "name": "Sridhar Vembu",
        "role": "CEO",
        "email": "sridhar@zoho.com",
        "email_confidence": 92
      },
      "outreach": {
        "subject": "AI transformation insights for Zoho",
        "body": "..."
      }
    }
  ]
}
```

## 🧪 Testing

Run in mock mode to test without consuming API credits:

```bash
python -m app.main --mock
```

This will:
- Use simulated RSS data
- Skip real Tavily searches
- Generate mock company/contact data
- Still run the full LLM pipeline

## 📈 Expected Results

Per run with default settings:
- 3 market trends detected
- ~9 companies found (3 per trend)
- ~18 contacts identified (2 per company)
- 40-60% email find rate
- ~10-15 complete leads with emails

## ⚡ Performance Tips

1. **Use Ollama** - Local inference = unlimited, fast
2. **Enable caching** - Same searches won't hit APIs twice
3. **Adjust limits** - Lower MAX_* values for testing
4. **Mock mode first** - Validate pipeline before real runs

## 🔒 API Rate Limits

| Service | Free Tier | Notes |
|---------|-----------|-------|
| Tavily | 1,000/month | ~18 full runs |
| Apollo.io | 600/month | Primary email finder |
| Hunter.io | 25/month | Fallback only |
| Gemini | 60 RPM | LLM fallback |
| Ollama | Unlimited | Local, recommended |

## 🐛 Troubleshooting

### Ollama not connecting
```bash
# Check Ollama is running
curl http://localhost:11434/api/tags

# Start Ollama if needed
ollama serve
```

### No emails found
- Check domain extraction in logs
- Verify Apollo/Hunter API keys
- Some Indian companies have poor email coverage

### LLM errors
- Ensure Ollama model is pulled: `ollama pull mistral`
- Check Gemini API key if Ollama fails

## 📝 License

MIT License - feel free to use and modify.

---

Built with ❤️ for Indian B2B sales teams
