"""
# seed_database.py

## What this script does:
This script creates and populates the SQLite database for the Velo
enterprise-knowledge-agent project. It does the following in order:

1. Creates all database tables (employees, departments, hr_requests)
2. Inserts the 4 guaranteed demo personas with specific roles and details
3. Generates ~70 additional realistic fake employees using the Faker library
4. Generates PTO and expense HR requests tied to each demo persona

Run this script once before starting the backend:
    python scripts/seed_database.py

The database will be saved to: internal_data/velo.db
"""

import os
import random
from datetime import datetime, timedelta
from pathlib import Path
from faker import Faker
from sqlalchemy import (
    create_engine, Column, Integer, String, Float,
    Date, Boolean, ForeignKey, Text, DateTime
)
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from dotenv import load_dotenv

env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

# Configuration

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./internal_data/velo.db")
fake = Faker()
Faker.seed(42)
random.seed(42)

os.makedirs("internal_data", exist_ok=True)

Base = declarative_base()
engine = create_engine(DATABASE_URL, echo=False)
Session = sessionmaker(bind=engine)

# Models

class Department(Base):
    __tablename__ = "departments"
    id            = Column(Integer, primary_key=True)
    name          = Column(String, unique=True, nullable=False)
    team_lead     = Column(String, nullable=False)
    headcount     = Column(Integer, default=0)
    budget_usd    = Column(Float, default=0)
    slack_channel = Column(String)
    employees     = relationship("Employee", back_populates="dept")


class Employee(Base):
    __tablename__ = "employees"
    id            = Column(Integer, primary_key=True)
    name          = Column(String, nullable=False)
    email         = Column(String, unique=True, nullable=False)
    role          = Column(String, nullable=False)
    department_id = Column(Integer, ForeignKey("departments.id"))
    salary_usd    = Column(Float)
    start_date    = Column(Date)
    is_manager    = Column(Boolean, default=False)
    reports_to    = Column(Integer, ForeignKey("employees.id"), nullable=True)
    persona       = Column(String, nullable=True)
    dept          = relationship("Department", back_populates="employees")
    hr_requests   = relationship("HRRequest", back_populates="employee")


class HRRequest(Base):
    __tablename__ = "hr_requests"
    id            = Column(Integer, primary_key=True)
    request_type  = Column(String)   # pto / expense
    description   = Column(Text)
    status        = Column(String)   # pending / approved / denied
    employee_id   = Column(Integer, ForeignKey("employees.id"))
    submitted_at  = Column(DateTime)
    resolved_at   = Column(DateTime, nullable=True)
    employee      = relationship("Employee", back_populates="hr_requests")


# Department Data

DEPARTMENTS = [
    {"name": "Engineering",      "team_lead": "Raj Mehta",     "budget_usd": 4500000, "slack_channel": "#engineering"},
    {"name": "Sales",            "team_lead": "Marcus Webb",   "budget_usd": 3200000, "slack_channel": "#sales"},
    {"name": "Customer Success", "team_lead": "Jordan Blake",  "budget_usd": 1800000, "slack_channel": "#cs"},
    {"name": "Operations",       "team_lead": "Priya Patel",   "budget_usd": 1200000, "slack_channel": "#operations"},
    {"name": "Marketing",        "team_lead": "Elena Vasquez", "budget_usd": 1500000, "slack_channel": "#marketing"},
    {"name": "Product",          "team_lead": "Danny Okafor",  "budget_usd": 2000000, "slack_channel": "#product"},
    {"name": "Finance",          "team_lead": "Rachel Kim",    "budget_usd": 900000,  "slack_channel": "#finance"},
    {"name": "Legal",            "team_lead": "Tom Bassett",   "budget_usd": 600000,  "slack_channel": "#legal"},
]

# Demo Personas

PERSONAS = [
    {
        "name":       "Sarah Chen",
        "email":      "sarah.chen@veloapp.io",
        "role":       "Junior Software Engineer",
        "department": "Engineering",
        "salary_usd": 95000,
        "start_date": datetime.today().date() - timedelta(days=5),
        "is_manager": False,
        "persona":    "new_hire",
    },
    {
        "name":       "Marcus Webb",
        "email":      "marcus.webb@veloapp.io",
        "role":       "Sales Manager",
        "department": "Sales",
        "salary_usd": 130000,
        "start_date": datetime.today().date() - timedelta(days=912),
        "is_manager": True,
        "persona":    "manager",
    },
    {
        "name":       "Priya Patel",
        "email":      "priya.patel@veloapp.io",
        "role":       "HR & Operations Lead",
        "department": "Operations",
        "salary_usd": 115000,
        "start_date": datetime.today().date() - timedelta(days=1095),
        "is_manager": True,
        "persona":    "ops",
    },
    {
        "name":       "Jordan Blake",
        "email":      "jordan.blake@veloapp.io",
        "role":       "VP of Customer Success",
        "department": "Customer Success",
        "salary_usd": 175000,
        "start_date": datetime.today().date() - timedelta(days=2190),
        "is_manager": True,
        "persona":    "exec",
    },
]

# Role Templates Per Department

ROLES = {
    "Engineering":      ["Software Engineer", "Senior Software Engineer", "Staff Engineer", "DevOps Engineer", "QA Engineer"],
    "Sales":            ["Account Executive", "Sales Development Rep", "Senior Account Executive", "Sales Engineer"],
    "Customer Success": ["Customer Success Manager", "Senior CSM", "Onboarding Specialist", "Renewals Manager"],
    "Operations":       ["Operations Manager", "HR Generalist", "IT Support Specialist", "Office Manager"],
    "Marketing":        ["Marketing Manager", "Content Strategist", "Demand Generation Manager", "Designer"],
    "Product":          ["Product Manager", "Senior Product Manager", "UX Designer", "Product Analyst"],
    "Finance":          ["Financial Analyst", "Senior Accountant", "FP&A Manager", "Controller"],
    "Legal":            ["Legal Counsel", "Contracts Manager", "Compliance Analyst"],
}

SALARY_RANGES = {
    "Engineering":      (90000,  180000),
    "Sales":            (70000,  150000),
    "Customer Success": (65000,  130000),
    "Operations":       (60000,  120000),
    "Marketing":        (65000,  125000),
    "Product":          (95000,  175000),
    "Finance":          (75000,  140000),
    "Legal":            (90000,  160000),
}

# Seed Functions

def seed_departments(session):
    dept_map = {}
    for d in DEPARTMENTS:
        dept = Department(
            name=d["name"],
            team_lead=d["team_lead"],
            budget_usd=d["budget_usd"],
            slack_channel=d["slack_channel"],
            headcount=0,
        )
        session.add(dept)
    session.commit()
    for dept in session.query(Department).all():
        dept_map[dept.name] = dept.id
    return dept_map


def seed_personas(session, dept_map):
    persona_map = {}
    for p in PERSONAS:
        emp = Employee(
            name=p["name"],
            email=p["email"],
            role=p["role"],
            department_id=dept_map[p["department"]],
            salary_usd=p["salary_usd"],
            start_date=p["start_date"],
            is_manager=p["is_manager"],
            persona=p["persona"],
        )
        session.add(emp)
        session.flush()
        persona_map[p["name"]] = emp.id
    session.commit()
    return persona_map


def seed_employees(session, dept_map, count=70):
    dept_names = list(ROLES.keys())
    created = 0
    for _ in range(count):
        dept_name = random.choice(dept_names)
        role      = random.choice(ROLES[dept_name])
        low, high = SALARY_RANGES[dept_name]
        start     = fake.date_between(start_date="-4y", end_date="-1m")
        emp = Employee(
            name=fake.name(),
            email=fake.unique.email(),
            role=role,
            department_id=dept_map[dept_name],
            salary_usd=round(random.uniform(low, high), -3),
            start_date=start,
            is_manager=random.random() < 0.15,
            persona=None,
        )
        session.add(emp)
        created += 1
    session.commit()
    for dept in session.query(Department).all():
        dept.headcount = session.query(Employee).filter_by(department_id=dept.id).count()
    session.commit()


def seed_hr_requests(session, persona_map):
    """
    Seeds PTO and expense requests for each persona.
    Each persona has a mix of approved and pending requests
    to show different statuses in the sidebar.
    """
    requests = [

        # Sarah Chen — new hire, just started
        {
            "type":        "pto",
            "description": "Requesting 3 days PTO June 2-4 for a family event. "
                           "First PTO request since joining the Engineering team.",
            "status":      "pending",
            "employee":    "Sarah Chen",
            "days_ago":    2,
        },
        {
            "type":        "expense",
            "description": "Home office setup — standing desk converter ($189) and "
                           "ergonomic keyboard ($95). Total $284 within the $500 "
                           "new hire home office stipend.",
            "status":      "approved",
            "employee":    "Sarah Chen",
            "days_ago":    4,
        },

        # Marcus Webb — Sales Manager, 2.5 years 
        {
            "type":        "pto",
            "description": "Requesting 5 days PTO April 21-25 for a family vacation. "
                           "Pipeline is healthy and the SDR team will cover inbound.",
            "status":      "approved",
            "employee":    "Marcus Webb",
            "days_ago":    14,
        },
        {
            "type":        "expense",
            "description": "Team dinner for Q1 pipeline review — $680 total for 9 "
                           "attendees at $75 per person. Requesting VP approval as "
                           "total exceeds the $500 manager threshold.",
            "status":      "approved",
            "employee":    "Marcus Webb",
            "days_ago":    5,
        },
        {
            "type":        "expense",
            "description": "SaaStr Annual 2024 conference registration — $1,200 "
                           "registration fee. Within the $1,500 conference limit "
                           "per the expense policy.",
            "status":      "pending",
            "employee":    "Marcus Webb",
            "days_ago":    1,
        },

        # Priya Patel — HR & Operations Lead, 3 years
        {
            "type":        "pto",
            "description": "Requesting 4 days PTO May 19-22 for a personal trip. "
                           "Onboarding for the April cohort will be complete by then.",
            "status":      "approved",
            "employee":    "Priya Patel",
            "days_ago":    20,
        },
        {
            "type":        "pto",
            "description": "Requesting 2 days PTO June 9-10 for a family commitment. "
                           "Coordinating coverage with the operations team.",
            "status":      "pending",
            "employee":    "Priya Patel",
            "days_ago":    1,
        },
        {
            "type":        "expense",
            "description": "SHRM Annual Conference registration — $1,350 for HR "
                           "leadership development. Within the $1,500 conference "
                           "limit. Directly relevant to the HR modernization initiative.",
            "status":      "approved",
            "employee":    "Priya Patel",
            "days_ago":    30,
        },

        # Jordan Blake — VP Customer Success, 6 years
        {
            "type":        "pto",
            "description": "Requesting 3 days PTO May 1-3 for a speaking engagement "
                           "at SaaS Connect conference in San Francisco. CS team "
                           "leadership will cover escalations.",
            "status":      "approved",
            "employee":    "Jordan Blake",
            "days_ago":    20,
        },
        {
            "type":        "pto",
            "description": "Requesting 1 week PTO July 14-18 for annual summer "
                           "vacation. Q2 renewals will be closed by then and the "
                           "team is fully staffed.",
            "status":      "pending",
            "employee":    "Jordan Blake",
            "days_ago":    3,
        },
        {
            "type":        "expense",
            "description": "Customer executive dinner in New York — 4 attendees, "
                           "$580 total. Requesting VP approval for above-limit "
                           "client entertainment expense.",
            "status":      "approved",
            "employee":    "Jordan Blake",
            "days_ago":    8,
        },
        {
            "type":        "expense",
            "description": "Flight and hotel for SaaS Connect speaking engagement "
                           "in San Francisco — $890 total (economy flight $340, "
                           "hotel 2 nights at $275/night).",
            "status":      "approved",
            "employee":    "Jordan Blake",
            "days_ago":    15,
        },
    ]

    today = datetime.now()
    for r in requests:
        emp_id = persona_map.get(r["employee"])
        hr_req = HRRequest(
            request_type=r["type"],
            description=r["description"],
            status=r["status"],
            employee_id=emp_id,
            submitted_at=today - timedelta(days=r["days_ago"]),
            resolved_at=today - timedelta(days=1) if r["status"] != "pending" else None,
        )
        session.add(hr_req)
    session.commit()

# Main
def main():
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)

    session = Session()
    try:
        dept_map    = seed_departments(session)
        persona_map = seed_personas(session, dept_map)
        seed_employees(session, dept_map, count=70)
        seed_hr_requests(session, persona_map)
        print("Database seeded successfully!")
    except Exception as e:
        session.rollback()
        print(f"Seed failed: {e}")
        raise
    finally:
        session.close()

if __name__ == "__main__":
    main()
