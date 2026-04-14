"""
# database.py

## What this file does:
Sets up the SQLAlchemy database connection and defines all table models
for the Velo enterprise-knowledge-agent backend. This file is the single
source of truth for the database schema in the backend.

Tables: departments, employees, hr_requests (pto and expense only)
Projects and IT tickets have been intentionally removed — the portal
focuses on HR knowledge and personal HR requests.

Imported by: agent.py, main.py
"""

import sys
from pathlib import Path
from sqlalchemy import (
    create_engine, Column, Integer, String,
    Float, Date, Boolean, ForeignKey, Text, DateTime,
    inspect, text
)
from sqlalchemy.orm import declarative_base, sessionmaker, relationship

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import config, validate_config

# ─── Engine & Session ──────────────────────────────────────────────────────────

engine = create_engine(
    config.DATABASE_URL,
    connect_args={"check_same_thread": False},
    echo=config.DEBUG,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# ─── Models ────────────────────────────────────────────────────────────────────

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


# ─── Dependencies ──────────────────────────────────────────────────────────────

def get_db():
    """FastAPI dependency — yields a database session and closes it after use."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_engine():
    """Returns the SQLAlchemy engine for use by the LangChain SQL agent tool."""
    return engine


# ─── Tests ─────────────────────────────────────────────────────────────────────

def run_tests():
    print("\n" + "="*55)
    print("  RUNNING DATABASE TESTS")
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

    # Database file exists
    db_path = Path(config.DATABASE_URL.replace("sqlite:///", ""))
    check("Database file exists", db_path.exists(), f"expected at {db_path}")

    # Engine connects
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        check("Engine connects successfully", True)
    except Exception as e:
        check("Engine connects successfully", False, str(e))

    # Tables exist
    inspector = inspect(engine)
    existing_tables = inspector.get_table_names()
    for table in ["departments", "employees", "hr_requests"]:
        check(f"Table '{table}' exists", table in existing_tables)

    # Removed tables should not exist
    for table in ["projects", "it_tickets"]:
        check(f"Table '{table}' correctly removed", table not in existing_tables)

    db = SessionLocal()
    try:
        # Row counts
        dept_count = db.query(Department).count()
        emp_count  = db.query(Employee).count()
        hr_count   = db.query(HRRequest).count()

        check("Departments table has rows",   dept_count > 0,  f"found {dept_count}")
        check("Employees table has rows",     emp_count > 0,   f"found {emp_count}")
        check("HR Requests table has rows",   hr_count > 0,    f"found {hr_count}")

        # All 4 personas queryable
        for name, persona in [
            ("Sarah Chen",  "new_hire"),
            ("Marcus Webb", "manager"),
            ("Priya Patel", "ops"),
            ("Jordan Blake","exec"),
        ]:
            emp = db.query(Employee).filter_by(name=name).first()
            check(f"Persona '{name}' is queryable",      emp is not None)
            check(f"Persona '{name}' has correct type",  emp and emp.persona == persona)
            check(f"Persona '{name}' has department",    emp and emp.dept is not None)

        # Relationships work
        sarah = db.query(Employee).filter_by(name="Sarah Chen").first()
        if sarah:
            check("Employee.dept relationship works",        sarah.dept is not None)
            check("Employee.hr_requests relationship works", sarah.hr_requests is not None)

        # Each persona has PTO and expense requests
        for name in ["Sarah Chen", "Marcus Webb", "Priya Patel", "Jordan Blake"]:
            emp = db.query(Employee).filter_by(name=name).first()
            if emp:
                pto = db.query(HRRequest).filter_by(employee_id=emp.id, request_type="pto").count()
                exp = db.query(HRRequest).filter_by(employee_id=emp.id, request_type="expense").count()
                check(f"{name} has PTO requests",     pto > 0, f"found {pto}")
                check(f"{name} has expense requests", exp > 0, f"found {exp}")

        # Raw SQL works for agent
        result = db.execute(text(
            "SELECT e.name, d.name as dept FROM employees e "
            "JOIN departments d ON e.department_id = d.id "
            "WHERE e.persona IS NOT NULL"
        )).fetchall()
        check("Raw SQL join query works", len(result) == 4, f"expected 4, got {len(result)}")

        check("get_engine() returns engine", get_engine() is not None)

    except Exception as e:
        check("Database queries ran without error", False, str(e))
    finally:
        db.close()

    print("="*55)
    print(f"  RESULTS: {passed} passed, {failed} failed")
    if failed == 0:
        print("  🎉 All database tests passed!")
    else:
        print("  ⚠️  Some tests failed — check output above")
    print("="*55 + "\n")

    return failed == 0


# ─── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "="*55)
    print("  VELO — Database Module")
    print("  enterprise-knowledge-agent")
    print("="*55)

    if not validate_config():
        sys.exit(1)

    success = run_tests()
    sys.exit(0 if success else 1)
    