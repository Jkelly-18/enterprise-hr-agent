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

# App Setup
if not validate_config():
    sys.exit(1)

app = FastAPI(
    title=config.APP_TITLE,
    version=config.APP_VERSION,
    description="AI-powered internal knowledge assistant for Velo employees",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request / Response Models
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

# Persona Definitions
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

# Routes
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

# Main
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
