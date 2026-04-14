"""
# agent.py

## What this file does:
This is the brain of the enterprise-knowledge-agent. It builds a LangGraph
ReAct agent that has access to two tools:

1. RAG Tool — searches the Velo company document library using ChromaDB
   vector search to find relevant policy, process, and onboarding information

2. SQL Tool — queries the Velo SQLite database to retrieve structured data
   about employees, departments, projects, IT tickets, and HR requests

The agent receives a user question plus the current user's context (name,
role, department, persona) and uses LangGraph's ReAct loop to decide which
tool(s) to use, in what order, and how to combine the results into a
coherent, personalized answer.

LangGraph replaces the older LangChain AgentExecutor with a more modern
graph-based approach that is more transparent, controllable, and production
ready.

Imported by: main.py
"""

import sys
from pathlib import Path

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langgraph.prebuilt import create_react_agent
from datetime import date

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import config, validate_config
from rag import retrieve_docs_for_role, get_role_from_persona
from database import get_engine

# ─── LLM ───────────────────────────────────────────────────────────────────────

llm = ChatOpenAI(
    model=config.OPENAI_MODEL,
    api_key=config.OPENAI_API_KEY,
    temperature=0,
    streaming=False,
)

# ─── System Prompt ─────────────────────────────────────────────────────────────

def build_system_prompt(
    user_name:       str,
    user_role:       str,
    user_department: str,
    user_persona:    str,
) -> str:
    return f"""You are Aria, the intelligent internal knowledge assistant for Velo — \
a fast-growing B2B SaaS company building marketing analytics software.

You help Velo employees instantly find answers about company policies, processes, \
onboarding steps, benefits, PTO, expense reporting, and team information.

You have access to two tools:
1. search_company_docs — searches Velo's internal document library (policies, handbooks, \
runbooks, playbooks, onboarding guides, expense policy, PTO policy)
2. query_employee_database — queries structured data about employees, departments, \
PTO requests, and expense reports

CURRENT USER CONTEXT:
Name: {user_name}
Role: {user_role}
Department: {user_department}
Persona: {user_persona}
Today's Date: {date.today().strftime("%B %d, %Y")}

INSTRUCTIONS:
- Always address the user by their first name
- Use the user's role and department to give role-specific answers

TENURE AND PTO RULES — ALWAYS FOLLOW THESE:
- For ANY question about PTO, vacation days, or time off allowance you MUST first query
  the employee database to get this user's exact start_date
- Calculate their tenure from start_date to today in years
- Then apply the correct tier: 0-2 years = 20 days, 2-5 years = 25 days, 5+ years = Unlimited
- Never list all three tiers — only tell the user their specific tier based on their tenure
- Always state how long they have been at Velo when answering PTO questions

ROLE-SPECIFIC ANSWER RULES:
- For onboarding and first 30/60/90 day questions, always retrieve the document that
  matches the user's specific department — Engineering gets the engineering plan,
  Sales gets the sales plan, Customer Success gets the CS plan
- For process questions (how to submit expenses, how to request PTO), give the
  step by step process from the relevant document — never just state the policy limit
- For expense questions, remind the user of the approval threshold relevant to their
  role level — managers have different limits than individual contributors
- For questions about the user's own PTO requests or expense reports, always query
  the database to show their actual submitted requests and their current status

TOOL USAGE RULES:
- Policy and process questions → search_company_docs
- Questions about the user's own requests, headcount, or specific employee data → query_employee_database
- Questions combining policy with personal data (e.g. my PTO allowance + my pending requests) → use both tools
- When querying for PTO tier or tenure, use: SELECT start_date FROM employees WHERE name = '{user_name}'
- The hr_requests table columns are: request_type (pto or expense), description, status (pending/approved/denied), submitted_at, employee_id
- Always cite the document name when answering from company docs
- Never make up information — only answer from what the tools return
- If you cannot find a definitive answer, suggest who to contact (HR: hr@veloapp.io, IT: #it-help)
- Be concise and direct — employees want quick answers not essays
- Format steps as numbered lists and bullet points for clarity"""


# ─── Tool Factory ──────────────────────────────────────────────────────────────

def build_tools(user_persona: str) -> list:
    """
    Builds and returns the list of tools for the agent.
    RAG tool is personalized to the user's role.
    SQL tools come from LangChain's SQLDatabaseToolkit.
    """
    role = get_role_from_persona(user_persona)

    # ── RAG Tool ──────────────────────────────────────────────────────────────
    # Using a closure to capture the role for this user session
    def _search_docs(query: str) -> str:
        """Search Velo's internal company documents for policies, processes,
        onboarding guides, HR policies, IT security, tools, OKRs, and procedures."""
        return retrieve_docs_for_role(query, role)

    # Manually create a tool with proper metadata for langgraph
    from langchain_core.tools import StructuredTool
    rag_tool = StructuredTool.from_function(
        func=_search_docs,
        name="search_company_docs",
        description=(
            "Search Velo's internal company documents. Use this for questions about: "
            "PTO policy and how to request time off in Rippling, expense policy and "
            "how to submit reports in Expensify, benefits, remote work policy, "
            "onboarding steps and 30/60/90 day plans by role, IT security, tools and "
            "software setup, engineering workflows, sales playbook, customer success "
            "runbook, company mission and OKRs, and any procedural or policy questions. "
            "Always use the role-specific onboarding document that matches the user's department."
        ),
    )

    # ── SQL Tools ─────────────────────────────────────────────────────────────
    db      = SQLDatabase(get_engine())
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    sql_tools = toolkit.get_tools()

    return [rag_tool] + sql_tools


# ─── Agent Builder ─────────────────────────────────────────────────────────────

def build_agent(
    user_name:       str = "Employee",
    user_role:       str = "Employee",
    user_department: str = "Velo",
    user_persona:    str = "new_hire",
):
    """
    Builds and returns a personalized LangGraph ReAct agent for the given user.
    Called once per conversation session with the user's profile data.

    LangGraph's create_react_agent creates a graph that loops:
    1. LLM decides what to do
    2. If tool call — execute the tool
    3. Feed result back to LLM
    4. Repeat until LLM returns a final answer

    Args:
        user_name:       Full name of the logged-in user
        user_role:       Job title of the user
        user_department: Department name of the user
        user_persona:    Persona type: new_hire / manager / ops / exec

    Returns:
        A compiled LangGraph agent ready to receive messages
    """
    tools  = build_tools(user_persona)
    prompt = build_system_prompt(user_name, user_role, user_department, user_persona)

    agent = create_react_agent(
        model=llm,
        tools=tools,
        prompt=prompt,
    )

    return agent


from langchain_core.messages import AIMessage

def ask(
    question:        str,
    user_name:       str = "Employee",
    user_role:       str = "Employee",
    user_department: str = "Velo",
    user_persona:    str = "new_hire",
    chat_history:    list = None,
) -> str:
    agent = build_agent(user_name, user_role, user_department, user_persona)

    # Convert chat history dicts to LangChain message objects
    messages = []
    for msg in (chat_history or []):
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            messages.append(AIMessage(content=msg["content"]))

    messages.append(HumanMessage(content=question))

    result = agent.invoke({"messages": messages})

    final_message = result["messages"][-1]
    if hasattr(final_message, "content"):
        return final_message.content
    return str(final_message)


# ─── Tests ─────────────────────────────────────────────────────────────────────

def run_tests():
    print("\n" + "="*55)
    print("  RUNNING AGENT TESTS")
    print("="*55)
    print("  Note: These tests make real OpenAI API calls")
    print("  and will take 30-60 seconds to complete.")
    print("="*55)

    passed = 0
    failed = 0

    def check(label, condition, detail=""):
        nonlocal passed, failed
        if condition:
            print(f"  ✅ {label}")
            passed += 1
        else:
            print(f"  ❌ FAILED: {label}" + (f" — {detail}" if detail else ""))
            failed += 1

    # Agent builds without error
    print("\n  Testing agent construction...")
    try:
        agent = build_agent(
            user_name="Sarah Chen",
            user_role="Junior Software Engineer",
            user_department="Engineering",
            user_persona="new_hire",
        )
        check("Agent builds successfully", True)
        check("Agent is not None",         agent is not None)
    except Exception as e:
        check("Agent builds successfully", False, str(e))
        print("  Cannot continue — agent failed to build")
        return False

    # Tools build correctly
    print("\n  Testing tool construction...")
    try:
        tools = build_tools("new_hire")
        check("Tools list is not empty",       len(tools) > 0)
        check("RAG tool present",              any("docs" in t.name for t in tools))
        check("SQL tool present",              any("sql" in t.name.lower() for t in tools))
        check("At least 2 tools available",    len(tools) >= 2)
    except Exception as e:
        check("Tools build successfully", False, str(e))

    # Real question answering tests
    test_cases = [
        {
            "label":        "New hire PTO question",
            "question":     "How many PTO days do I get as a new employee?",
            "persona":      "new_hire",
            "name":         "Sarah Chen",
            "role":         "Junior Software Engineer",
            "dept":         "Engineering",
            "must_contain": ["20", "pto"],
        },
        {
            "label":        "Engineering setup question",
            "question":     "What do I need to set up on my first day as an engineer?",
            "persona":      "new_hire",
            "name":         "Sarah Chen",
            "role":         "Junior Software Engineer",
            "dept":         "Engineering",
            "must_contain": ["github", "macbook"],
        },
        {
            "label":        "SQL headcount question",
            "question":     "How many people are in the Engineering department?",
            "persona":      "manager",
            "name":         "Marcus Webb",
            "role":         "Sales Manager",
            "dept":         "Sales",
            "must_contain": ["engineering"],
        },
        {
            "label":        "Expense policy question",
            "question":     "What is the meal expense limit per person?",
            "persona":      "ops",
            "name":         "Priya Patel",
            "role":         "HR & Operations Lead",
            "dept":         "Operations",
            "must_contain": ["75"],
        },
    ]

    print("\n  Testing agent question answering...")
    for test in test_cases:
        print(f"\n  🤔 Asking: '{test['question']}'")
        try:
            answer = ask(
                question=test["question"],
                user_name=test["name"],
                user_role=test["role"],
                user_department=test["dept"],
                user_persona=test["persona"],
            )
            print(f"  💬 Answer preview: {answer[:200]}...")
            check(
                f"{test['label']} — returns an answer",
                bool(answer) and len(answer) > 20,
                f"got {len(answer)} chars"
            )
            for keyword in test["must_contain"]:
                check(
                    f"{test['label']} — contains '{keyword}'",
                    keyword.lower() in answer.lower(),
                    f"keyword '{keyword}' not found"
                )
        except Exception as e:
            check(f"{test['label']} — runs without error", False, str(e))

    print("\n" + "="*55)
    print(f"  RESULTS: {passed} passed, {failed} failed")
    if failed == 0:
        print("  🎉 All agent tests passed!")
    else:
        print("  ⚠️  Some tests failed — check output above")
    print("="*55 + "\n")

    return failed == 0


# ─── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "="*55)
    print("  VELO — Agent Module (LangGraph)")
    print("  enterprise-knowledge-agent")
    print("="*55)

    if not validate_config():
        sys.exit(1)

    success = run_tests()
    sys.exit(0 if success else 1)
    