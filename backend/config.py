"""
# config.py

## What this file does:
Central configuration module for the enterprise-knowledge-agent backend.
Loads all environment variables from the .env file and makes them available
as a single config object that every other backend module imports from.

Also validates that all required variables are present on startup so the
app fails fast with a clear error message rather than breaking mysteriously
later when a variable is actually used.

Imported by: database.py, rag.py, agent.py, main.py
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# ─── Load .env ─────────────────────────────────────────────────────────────────
# Explicitly resolve path to .env in project root so this works regardless
# of which directory the app is started from

env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

# ─── Config Values ─────────────────────────────────────────────────────────────

class Config:
    # OpenAI
    OPENAI_API_KEY:     str  = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL:       str  = os.getenv("OPENAI_MODEL", "gpt-4o")
    EMBEDDING_MODEL:    str  = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

    # Database
    DATABASE_URL:       str  = os.getenv("DATABASE_URL", "sqlite:///./internal_data/velo.db")

    # ChromaDB
    CHROMA_PATH:        str  = os.getenv("CHROMA_PATH", "./internal_data/chroma")
    COLLECTION_NAME:    str  = "velo_knowledge_base"

    # RAG settings
    RAG_TOP_K:          int  = int(os.getenv("RAG_TOP_K", "5"))

    # App settings
    APP_TITLE:          str  = "Velo Enterprise Knowledge Agent"
    APP_VERSION:        str  = "1.0.0"
    DEBUG:              bool = os.getenv("DEBUG", "false").lower() == "true"

    # CORS — frontend origin (update when deploying)
    FRONTEND_URL:       str  = os.getenv("FRONTEND_URL", "http://localhost:3000")

    # Required variables — app will not start without these
    REQUIRED = ["OPENAI_API_KEY", "DATABASE_URL", "CHROMA_PATH"]

config = Config()

# ─── Validation ────────────────────────────────────────────────────────────────

def validate_config() -> bool:
    """
    Validates all required environment variables are present and non-empty.
    Called on app startup. Returns True if valid, exits with error if not.
    """
    missing = []
    for key in Config.REQUIRED:
        value = getattr(config, key, None)
        if not value:
            missing.append(key)

    if missing:
        print("\n❌ Missing required environment variables:")
        for key in missing:
            print(f"   - {key}")
        print("\nMake sure your .env file exists in the project root and contains all required variables.")
        print("See .env.example for reference.\n")
        return False

    return True


# ─── Tests ─────────────────────────────────────────────────────────────────────

def run_tests():
    print("\n" + "="*55)
    print("  RUNNING CONFIG TESTS")
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

    # Required fields present
    check("OPENAI_API_KEY is set",      bool(config.OPENAI_API_KEY),  "not found in .env")
    check("DATABASE_URL is set",        bool(config.DATABASE_URL),    "not found in .env")
    check("CHROMA_PATH is set",         bool(config.CHROMA_PATH),     "not found in .env")

    # Values are correct types
    check("RAG_TOP_K is an integer",    isinstance(config.RAG_TOP_K, int),  f"got {type(config.RAG_TOP_K)}")
    check("DEBUG is a boolean",         isinstance(config.DEBUG, bool),     f"got {type(config.DEBUG)}")

    # Defaults are sensible
    check("OPENAI_MODEL is set",        bool(config.OPENAI_MODEL),    f"got '{config.OPENAI_MODEL}'")
    check("EMBEDDING_MODEL is set",     bool(config.EMBEDDING_MODEL), f"got '{config.EMBEDDING_MODEL}'")
    check("COLLECTION_NAME is set",     bool(config.COLLECTION_NAME), f"got '{config.COLLECTION_NAME}'")
    check("APP_TITLE is set",           bool(config.APP_TITLE),       f"got '{config.APP_TITLE}'")
    check("FRONTEND_URL is set",        bool(config.FRONTEND_URL),    f"got '{config.FRONTEND_URL}'")

    # RAG_TOP_K is a reasonable value
    check("RAG_TOP_K is between 1-20",  1 <= config.RAG_TOP_K <= 20, f"got {config.RAG_TOP_K}")

    # API key format check (starts with sk-)
    check(
        "OPENAI_API_KEY has valid format",
        config.OPENAI_API_KEY.startswith("sk-"),
        "key should start with 'sk-'"
    )

    # Database URL points to correct file
    check(
        "DATABASE_URL points to velo.db",
        "velo.db" in config.DATABASE_URL,
        f"got '{config.DATABASE_URL}'"
    )

    # ChromaDB path points to internal_data
    check(
        "CHROMA_PATH points to internal_data",
        "internal_data" in config.CHROMA_PATH,
        f"got '{config.CHROMA_PATH}'"
    )

    # validate_config() passes
    check("validate_config() returns True", validate_config())

    print("="*55)
    print(f"  RESULTS: {passed} passed, {failed} failed")
    if failed == 0:
        print("  🎉 All config tests passed!")
    else:
        print("  ⚠️  Some tests failed — check your .env file")
    print("="*55 + "\n")

    return failed == 0


# ─── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "="*55)
    print("  VELO — Config Module")
    print("  enterprise-knowledge-agent")
    print("="*55)
    print("\n📋 Loaded configuration:")
    print(f"  OPENAI_MODEL:     {config.OPENAI_MODEL}")
    print(f"  EMBEDDING_MODEL:  {config.EMBEDDING_MODEL}")
    print(f"  DATABASE_URL:     {config.DATABASE_URL}")
    print(f"  CHROMA_PATH:      {config.CHROMA_PATH}")
    print(f"  COLLECTION_NAME:  {config.COLLECTION_NAME}")
    print(f"  RAG_TOP_K:        {config.RAG_TOP_K}")
    print(f"  DEBUG:            {config.DEBUG}")
    print(f"  FRONTEND_URL:     {config.FRONTEND_URL}")
    print(f"  OPENAI_API_KEY:   {config.OPENAI_API_KEY[:7]}...{config.OPENAI_API_KEY[-4:]}")

    success = run_tests()
    sys.exit(0 if success else 1)
