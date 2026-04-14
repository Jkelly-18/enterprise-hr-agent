# Enterprise HR Agent

An AI-powered enterprise HR assistant built to demonstrate how a modern agent stack can solve real operational problems at scale. The application combines a RAG pipeline, a LangGraph reasoning agent, vector search, and a relational database into a single full stack system deployed in a production environment.

**Live Demo:** https://enterprise-agent.netlify.app

---

## Overview

Enterprise HR Agent is a portfolio project built around a fictional B2B SaaS company called Velo. With 74 employees across 8 departments and a rapidly growing headcount, Velo faces a problem common to fast-scaling companies: internal knowledge exists but is not easily accessible. New employees repeatedly ask managers the same questions. HR teams handle constant streams of basic requests. Onboarding is slower and more expensive than it needs to be.

The solution is Aria, an AI-powered HR agent that gives every employee instant, personalized answers grounded in real company documents and live employee data. Employees ask questions in plain language. Aria determines whether the answer lives in a policy document or in the employee database, queries the right source or both simultaneously, and returns a single accurate answer tailored to who is asking.

A new engineer asking about their first 30 days gets the engineering onboarding plan. A six year VP asking about PTO gets told they are on the unlimited tier based on their actual start date pulled directly from the database. Every answer reflects the individual, not generic policy text.

The portal also displays each employee's active PTO requests and expense reports directly in the sidebar, giving employees a single place to both ask questions and track the status of their own HR activity in real time.

---

## How It Works

### RAG Pipeline

13 internal HR documents are chunked into 800 character overlapping segments using LangChain's RecursiveCharacterTextSplitter with 150 character overlap to preserve context at chunk boundaries. Each chunk is embedded using OpenAI's text-embedding-3-small model and stored in ChromaDB with metadata including document category and relevant employee role. At query time the user's question is embedded and the top 5 most semantically similar chunks are retrieved as grounding context for the LLM.

### LangGraph ReAct Agent

The agent is built on LangGraph's ReAct architecture. On each turn the LLM reasons about what information it needs, selects a tool, executes it, observes the result, and decides whether to call another tool or synthesize a final answer. This loop allows the agent to combine results from multiple sources in a single response rather than following a fixed retrieval path.

The agent has access to two tools:

- **search_company_docs:** semantic search over the ChromaDB vector store, filtered by employee role metadata when relevant
- **query_employee_database:** natural language to SQL queries via LangChain's SQLDatabaseToolkit, giving the agent access to live employee records, department structure, and HR requests

### Dual Data Sources

Unstructured knowledge lives in ChromaDB as vector embeddings. Structured employee data lives in SQLite via SQLAlchemy and includes three tables: employees, departments, and hr_requests. The agent treats each as a separate tool and can query both within the same conversation turn, for example retrieving a PTO policy from documents while simultaneously looking up the user's exact start date from the database to calculate their specific tenure tier.

### Personalized System Prompt

Every agent session is initialized with a dynamically built system prompt containing the current user's name, role, department, persona type, and today's date. The prompt includes explicit rules for tenure based PTO calculation, role specific document retrieval, and SQL query patterns. This ensures every answer is grounded in who is asking rather than returning generic policy text.

### Conversation Memory

Chat history is maintained on the frontend and sent with every request so the agent can reference previous messages and build on earlier answers across a multi-turn conversation.

---

## Company Documents

The 13 internal Velo HR documents were generated using Claude as a writing tool to produce realistic, consistently branded company content. All documents are formatted as PDFs and total approximately 55 pages of content across the following categories:

**General Onboarding (1 document, ~5 pages)**
New hire onboarding guide covering first day checklist, account setup, key contacts, company values, and new hire FAQ.

**Role-Specific Onboarding (3 documents, ~10 pages)**
Separate 30/60/90 day plans for Engineering, Sales, and Customer Success. Each plan defines success milestones, key contacts, and tool setup specific to that role. These documents are what make role-based retrieval meaningful — an engineer and a sales rep get completely different onboarding answers.

**HR Policy (3 documents, ~12 pages)**
Employee handbook covering PTO tiers by tenure, sick leave, remote work policy, performance reviews, and offboarding. Benefits guide covering health insurance, 401k matching, parental leave, and learning stipends. Expense and reimbursement policy covering submission process in Expensify, approval thresholds by role level, and common mistakes to avoid.

**IT and Security (2 documents, ~8 pages)**
IT security policy covering device requirements, password standards, data classification tiers, and incident reporting. Tech stack and tools guide covering every software platform the company uses across Sales, Engineering, HR, and Finance with access instructions.

**Internal Processes (3 documents, ~14 pages)**
Engineering team processes covering sprint structure, ticket lifecycle, code review standards, and deployment process. Sales playbook covering ICP definition, six stage pipeline, MEDDIC discovery framework, objection handling, and deal desk approvals. Customer success runbook covering account health scoring, onboarding process, QBR structure, escalation tiers, and churn risk signals.

**Company (1 document, ~6 pages)**
Company mission, core values, OKR framework, and current quarter company OKRs with key results.

---

## Employee Database

The SQLite database is seeded on every deployment by `scripts/seed_database.py` using the Faker library for bulk employee generation and manually written data for the four demo personas. The database contains:

- **74 employees** across 8 departments with realistic roles, salaries, and start dates
- **8 departments** with team leads, budgets, and Slack channels
- **12 HR requests** covering a mix of PTO requests and expense reports tied to each demo persona with pending and approved statuses

The four demo personas are hardcoded with specific start dates to cover all three PTO tiers: Sarah Chen at 5 days tenure (20 days PTO), Marcus Webb at 2.5 years (25 days), Priya Patel at 3 years (25 days), and Jordan Blake at 6 years (unlimited).

---

## Demo Personas

The portal includes four employee profiles that can be switched using the Switch Employee button. Each persona has a different role, department, and tenure, allowing the same question to produce meaningfully different answers depending on who is asking.

| Name | Role | Department | Tenure |
|---|---|---|---|
| Sarah Chen | Junior Software Engineer | Engineering | 5 days |
| Marcus Webb | Sales Manager | Sales | 2.5 years |
| Priya Patel | HR & Operations Lead | Operations | 3 years |
| Jordan Blake | VP of Customer Success | Customer Success | 6 years |

Try asking the same question as each persona to see how the agent personalizes its answer based on role, department, and tenure data from the database.

---

## Tech Stack

| Layer | Technology |
|---|---|
| LLM | OpenAI GPT-4o |
| Embeddings | OpenAI text-embedding-3-small |
| Agent Framework | LangGraph ReAct |
| RAG / Vector Search | ChromaDB |
| Document Processing | pypdf, LangChain text splitters |
| SQL ORM | SQLAlchemy |
| Database | SQLite |
| Backend API | FastAPI |
| Rate Limiting | slowapi |
| Frontend | React, Vite |
| HTTP Client | axios |
| Backend Deployment | Railway |
| Frontend Deployment | Netlify |

---

## Deployment Architecture

**Backend: Railway**

The FastAPI backend is deployed on Railway and connected directly to this GitHub repository. Railway automatically redeploys on every push to the main branch. On each deployment the startup command runs `seed_database.py` to recreate the SQLite database and `ingest_docs.py` to re-embed all 13 documents into ChromaDB before starting the uvicorn server. This ensures the database and vector store are always fresh without needing to commit binary files to the repository.

The API is rate limited using slowapi. The chat endpoint is limited to 10 requests per minute per IP address and all other endpoints are limited to 30 requests per minute.

**Frontend: Netlify**

The React frontend is built with Vite and deployed to Netlify as a static site. The compiled `dist/` folder is uploaded to Netlify. The frontend communicates with the Railway backend via axios HTTP requests. All API keys and secrets are stored as environment variables on Railway and never exposed to the browser.

---

## Project Structure

```
enterprise-knowledge-agent/
├── backend/
│   ├── agent.py            # LangGraph ReAct agent, system prompt, tool definitions
│   ├── config.py           # Environment variable loading and validation
│   ├── database.py         # SQLAlchemy models and database connection
│   ├── main.py             # FastAPI app, endpoints, rate limiting
│   └── rag.py              # ChromaDB connection and retrieval functions
├── company_docs/
│   ├── 01_new_hire_onboarding_guide.pdf
│   ├── 02_engineering_30_60_90_plan.pdf
│   ├── 03_sales_30_60_90_plan.pdf
│   ├── 04_customer_success_30_60_90_plan.pdf
│   ├── 05_employee_handbook.pdf
│   ├── 06_benefits_guide.pdf
│   ├── 07_expense_reimbursement_policy.pdf
│   ├── 08_it_security_policy.pdf
│   ├── 09_tech_stack_tools_guide.pdf
│   ├── 10_engineering_team_processes.pdf
│   ├── 11_sales_playbook_overview.pdf
│   ├── 12_customer_success_runbook.pdf
│   └── 13_company_mission_okr_framework.pdf
├── scripts/
│   ├── seed_database.py    # Creates and populates SQLite database
│   └── ingest_docs.py      # Chunks, embeds, and loads PDFs into ChromaDB
├── src/
│   ├── App.jsx             # React frontend, all components
│   └── App.css             # Styles
├── .gitignore              
├── eslint.config.js        # JavaScript linting configuration
├── index.html              # React app entry point
├── package.json            # Node dependencies and build scripts
├── package-lock.json       # Locked Node dependency versions
├── requirements.txt        # Python dependencies
└── vite.config.js          # Vite build configuration
```

---

## Running Locally

**Prerequisites:** Python 3.11+, Node.js 18+, OpenAI API key

**1. Clone the repository**
```bash
git clone https://github.com/Jkelly-18/enterprise-knowledge-agent.git
cd enterprise-knowledge-agent
```

**2. Create and activate a virtual environment**

The virtual environment isolates Python dependencies for this project.
```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux
```

**3. Install Python dependencies**
```bash
pip install -r requirements.txt
```

**4. Create your .env file**

The `.env` file is not included in the repository for security reasons.
Create it manually in the root directory with the following contents:

```
OPENAI_API_KEY=your_key_here
DATABASE_URL=sqlite:///./internal_data/velo.db
CHROMA_PATH=./internal_data/chroma
```

You can get an OpenAI API key at platform.openai.com. The database and
ChromaDB paths are created automatically in the next step.

**5. Seed the database and ingest documents**

These scripts generate the SQLite database and load all 13 documents
into ChromaDB. They must be run before starting the backend.
```bash
python scripts/seed_database.py
python scripts/ingest_docs.py
```

**6. Start the backend**
```bash
uvicorn backend.main:app --reload --port 8000
```

**7. Install Node dependencies and start the frontend**
```bash
npm install
npm run dev
```

Open `http://localhost:5173` in your browser.
