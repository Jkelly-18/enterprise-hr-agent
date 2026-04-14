"""
# main.py

## What this file does:
The FastAPI application that exposes the enterprise-knowledge-agent as a
REST API. This is the layer between the React frontend and the LangChain
agent. Every message from the chat UI hits an endpoint here, gets routed
to the agent with user context, and returns the agent's answer.

Endpoints:
- GET  /                          — health check
- GET  /api/personas              — returns the 4 demo personas
- GET  /api/user/{persona_id}     — returns full user profile
- POST /api/chat                  — main chat endpoint
- GET  /api/hr_requests/{persona} — returns PTO and expense requests

Run the server with:
    uvicorn backend.main:app --reload --port 8000
"""

import sys
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy.orm import Session

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import config, validate_config
from database import get_db, Employee, Department, HRRequest
from agent import ask

# ─── App Setup ─────────────────────────────────────────────────────────────────

if not validate_config():
    sys.exit(1)

app = FastAPI(
    title=config.APP_TITLE,
    version=config.APP_VERSION,
    description="AI-powered internal knowledge assistant for Velo employees",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[config.FRONTEND_URL, "http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Request / Response Models ──────────────────────────────────────────────────

class ChatRequest(BaseModel):
    question:     str
    persona:      str
    user_name:    Optional[str] = "Employee"
    chat_history: Optional[list] = []

class ChatResponse(BaseModel):
    answer:    str
    persona:   str
    user_name: str

class PersonaProfile(BaseModel):
    id:         str
    name:       str
    role:       str
    department: str
    persona:    str
    tagline:    str

# ─── Persona Definitions ───────────────────────────────────────────────────────

PERSONAS = [
    PersonaProfile(
        id="new_hire",
        name="Sarah Chen",
        role="Junior Software Engineer",
        department="Engineering",
        persona="new_hire",
        tagline="Just joined — figuring out the ropes",
    ),
    PersonaProfile(
        id="manager",
        name="Marcus Webb",
        role="Sales Manager",
        department="Sales",
        persona="manager",
        tagline="2.5 years in — leading the sales team",
    ),
    PersonaProfile(
        id="ops",
        name="Priya Patel",
        role="HR & Operations Lead",
        department="Operations",
        persona="ops",
        tagline="3 years in — keeps everything running",
    ),
    PersonaProfile(
        id="exec",
        name="Jordan Blake",
        role="VP of Customer Success",
        department="Customer Success",
        persona="exec",
        tagline="6 years in — owns the customer relationship",
    ),
]

PERSONA_MAP = {p.id: p for p in PERSONAS}

# ─── Routes ────────────────────────────────────────────────────────────────────

@app.get("/")
def health_check():
    return {
        "status":  "ok",
        "app":     config.APP_TITLE,
        "version": config.APP_VERSION,
        "message": "Velo Enterprise Knowledge Agent is running",
    }


@app.get("/api/personas")
def get_personas():
    return {"personas": PERSONAS}


@app.get("/api/user/{persona_id}")
def get_user_profile(persona_id: str, db: Session = Depends(get_db)):
    if persona_id not in PERSONA_MAP:
        raise HTTPException(status_code=404, detail=f"Persona '{persona_id}' not found")

    persona  = PERSONA_MAP[persona_id]
    employee = db.query(Employee).filter_by(name=persona.name).first()

    if not employee:
        raise HTTPException(status_code=404, detail=f"Employee '{persona.name}' not found")

    dept = db.query(Department).filter_by(id=employee.department_id).first()

    return {
        "id":         persona_id,
        "name":       employee.name,
        "email":      employee.email,
        "role":       employee.role,
        "department": dept.name if dept else "Unknown",
        "start_date": str(employee.start_date),
        "is_manager": employee.is_manager,
        "persona":    employee.persona,
        "tagline":    persona.tagline,
    }


@app.post("/api/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    if request.persona not in PERSONA_MAP:
        raise HTTPException(status_code=400, detail=f"Invalid persona: {request.persona}")

    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    persona = PERSONA_MAP[request.persona]

    answer = ask(
        question=request.question,
        user_name=persona.name,
        user_role=persona.role,
        user_department=persona.department,
        user_persona=persona.id,
        chat_history=request.chat_history or [],
    )

    return ChatResponse(
        answer=answer,
        persona=request.persona,
        user_name=persona.name,
    )


@app.get("/api/hr_requests/{persona_id}")
def get_hr_requests(persona_id: str, db: Session = Depends(get_db)):
    """
    Returns all PTO and expense requests for a given persona.
    Split into two lists so the frontend can display them separately.
    """
    if persona_id not in PERSONA_MAP:
        raise HTTPException(status_code=404, detail=f"Persona '{persona_id}' not found")

    persona  = PERSONA_MAP[persona_id]
    employee = db.query(Employee).filter_by(name=persona.name).first()

    if not employee:
        return {"pto_requests": [], "expense_requests": []}

    all_requests = db.query(HRRequest).filter_by(employee_id=employee.id).all()

    def format_request(r):
        return {
            "id":           r.id,
            "request_type": r.request_type,
            "description":  r.description,
            "status":       r.status,
            "submitted_at": str(r.submitted_at)[:10],
        }

    pto_requests     = [format_request(r) for r in all_requests if r.request_type == "pto"]
    expense_requests = [format_request(r) for r in all_requests if r.request_type == "expense"]

    return {
        "pto_requests":     pto_requests,
        "expense_requests": expense_requests,
    }


# ─── Tests ─────────────────────────────────────────────────────────────────────

def run_tests():
    from fastapi.testclient import TestClient

    print("\n" + "="*55)
    print("  RUNNING API TESTS")
    print("="*55)

    passed = 0
    failed = 0
    client = TestClient(app)

    def check(label, condition, detail=""):
        nonlocal passed, failed
        if condition:
            print(f"  ✅ {label}")
            passed += 1
        else:
            print(f"  ❌ FAILED: {label}" + (f" — {detail}" if detail else ""))
            failed += 1

    # Health check
    print("\n  Testing health check...")
    r = client.get("/")
    check("GET / returns 200",          r.status_code == 200)
    check("GET / returns status ok",    r.json().get("status") == "ok")

    # Personas
    print("\n  Testing personas endpoint...")
    r = client.get("/api/personas")
    check("GET /api/personas returns 200", r.status_code == 200)
    data = r.json()
    check("Returns 4 personas",            len(data.get("personas", [])) == 4)
    persona_ids = [p["id"] for p in data.get("personas", [])]
    for pid in ["new_hire", "manager", "ops", "exec"]:
        check(f"Persona '{pid}' present", pid in persona_ids)

    # User profiles
    print("\n  Testing user profile endpoint...")
    for pid in ["new_hire", "manager", "ops", "exec"]:
        r = client.get(f"/api/user/{pid}")
        check(f"GET /api/user/{pid} returns 200", r.status_code == 200)
        data = r.json()
        check(f"User '{pid}' has name",       bool(data.get("name")))
        check(f"User '{pid}' has email",      bool(data.get("email")))
        check(f"User '{pid}' has start_date", bool(data.get("start_date")))

    r = client.get("/api/user/invalid_persona")
    check("Invalid persona returns 404", r.status_code == 404)

    # HR requests endpoint
    print("\n  Testing HR requests endpoint...")
    for pid in ["new_hire", "manager", "ops", "exec"]:
        r = client.get(f"/api/hr_requests/{pid}")
        check(f"GET /api/hr_requests/{pid} returns 200", r.status_code == 200)
        data = r.json()
        check("Response has pto_requests key",     "pto_requests" in data)
        check("Response has expense_requests key", "expense_requests" in data)
        check(f"{pid} has PTO requests",            len(data.get("pto_requests", [])) > 0)
        check(f"{pid} has expense requests",        len(data.get("expense_requests", [])) > 0)

    # Verify old endpoints are gone
    print("\n  Verifying removed endpoints...")
    r = client.get("/api/tickets/new_hire")
    check("Old /api/tickets endpoint removed", r.status_code == 404)
    r = client.get("/api/projects/new_hire")
    check("Old /api/projects endpoint removed", r.status_code == 404)

    # Chat validation
    print("\n  Testing chat endpoint validation...")
    r = client.post("/api/chat", json={"question": "", "persona": "new_hire"})
    check("Empty question returns 400",       r.status_code in [400, 422])
    r = client.post("/api/chat", json={"question": "Hello", "persona": "invalid"})
    check("Invalid persona returns 400",      r.status_code == 400)

    print("\n" + "="*55)
    print(f"  RESULTS: {passed} passed, {failed} failed")
    if failed == 0:
        print("  🎉 All API tests passed!")
    else:
        print("  ⚠️  Some tests failed — check output above")
    print("="*55 + "\n")

    return failed == 0


# ─── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    print("\n" + "="*55)
    print("  VELO — API Module")
    print("  enterprise-knowledge-agent")
    print("="*55)

    if "--test" in sys.argv:
        success = run_tests()
        sys.exit(0 if success else 1)

    print("\n🚀 Starting Velo API server...")
    print("   Docs:   http://localhost:8000/docs")
    print("   Health: http://localhost:8000/\n")

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
    