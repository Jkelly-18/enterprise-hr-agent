"""
# rag.py

## What this file does:
Manages the connection to ChromaDB and provides retrieval functions
that the LangChain agent uses as its RAG tool. When the agent needs
to search company documents it calls the functions in this file.

Key functions:
- get_collection(): returns the ChromaDB collection
- retrieve_docs(): takes a query string and returns the most relevant
  document chunks with their source metadata
- retrieve_docs_for_role(): same as above but filtered by employee role
  so engineering questions prioritize engineering docs

Imported by: agent.py
"""

import sys
from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import config, validate_config

# ─── ChromaDB Client ───────────────────────────────────────────────────────────

embedding_fn = OpenAIEmbeddingFunction(
    api_key=config.OPENAI_API_KEY,
    model_name=config.EMBEDDING_MODEL,
)

chroma_client = chromadb.PersistentClient(path=config.CHROMA_PATH)


def get_collection():
    """Returns the Velo knowledge base ChromaDB collection."""
    return chroma_client.get_collection(
        name=config.COLLECTION_NAME,
        embedding_function=embedding_fn,
    )


# ─── Retrieval Functions ───────────────────────────────────────────────────────

def retrieve_docs(query: str, n_results: int = None) -> str:
    """
    Retrieves the most semantically relevant document chunks for a query.
    Returns a formatted string the LangChain agent can read directly.

    Args:
        query:     Natural language question or search string
        n_results: Number of chunks to retrieve (defaults to RAG_TOP_K in config)

    Returns:
        Formatted string of relevant document excerpts with source labels
    """
    if n_results is None:
        n_results = config.RAG_TOP_K

    collection = get_collection()
    results = collection.query(
        query_texts=[query],
        n_results=n_results,
    )

    if not results or not results["documents"][0]:
        return "No relevant documents found for this query."

    formatted = []
    for i, (doc, meta) in enumerate(
        zip(results["documents"][0], results["metadatas"][0])
    ):
        source   = meta.get("source", "Unknown")
        doc_name = meta.get("doc_name", source)
        formatted.append(
            f"[Source: {doc_name}]\n{doc}"
        )

    return "\n\n---\n\n".join(formatted)


def retrieve_docs_for_role(query: str, role: str, n_results: int = None) -> str:
    """
    Retrieves relevant document chunks filtered by employee role.
    First tries role-specific docs, falls back to all docs if needed.

    Args:
        query:     Natural language question
        role:      Employee role — 'engineering', 'sales', 'customer_success', 'all'
        n_results: Number of chunks to retrieve

    Returns:
        Formatted string of relevant document excerpts with source labels
    """
    if n_results is None:
        n_results = config.RAG_TOP_K

    collection = get_collection()

    # Try role-specific retrieval first
    if role and role != "all":
        try:
            results = collection.query(
                query_texts=[query],
                n_results=n_results,
                where={"relevant_to": {"$in": [role, "all"]}},
            )
            if results and results["documents"][0]:
                formatted = []
                for i, (doc, meta) in enumerate(
                    zip(results["documents"][0], results["metadatas"][0])
                ):
                    source   = meta.get("source", "Unknown")
                    doc_name = meta.get("doc_name", source)
                    formatted.append(f"[Source: {doc_name}]\n{doc}")
                return "\n\n---\n\n".join(formatted)
        except Exception:
            pass

    # Fall back to unfiltered retrieval
    return retrieve_docs(query, n_results)


def get_role_from_persona(persona: str) -> str:
    """Maps a persona type to a ChromaDB role filter value."""
    mapping = {
        "new_hire": "all",
        "manager":  "sales",
        "ops":      "all",
        "exec":     "customer_success",
    }
    return mapping.get(persona, "all")


# ─── Tests ─────────────────────────────────────────────────────────────────────

def run_tests():
    print("\n" + "="*55)
    print("  RUNNING RAG TESTS")
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

    # Collection loads successfully
    try:
        collection = get_collection()
        count = collection.count()
        check("ChromaDB collection loads",         True)
        check("Collection has documents",          count > 0,  f"found {count} chunks")
        check("Collection has at least 50 chunks", count >= 50, f"found {count}")
    except Exception as e:
        check("ChromaDB collection loads", False, str(e))
        print("="*55)
        print("  ❌ Cannot continue — ChromaDB not available")
        print("  Run scripts/ingest_docs.py first")
        print("="*55)
        return False

    # retrieve_docs returns results
    result = retrieve_docs("What is the PTO policy?")
    check("retrieve_docs returns results",         bool(result))
    check("retrieve_docs result is a string",      isinstance(result, str))
    check("retrieve_docs includes source label",   "[Source" in result)
    check("retrieve_docs not empty",               len(result) > 100, f"got {len(result)} chars")

    # retrieve_docs finds correct documents
    retrieval_tests = [
        ("What is the PTO policy?",                  "handbook",      "PTO query"),
        ("How do I set up my dev environment?",      "engineering",   "Dev setup query"),
        ("What are the expense reimbursement limits?","expense",       "Expense query"),
        ("What tools does the sales team use?",      "tech",          "Sales tools query"),
        ("How does the CS escalation process work?", "customer",      "CS escalation query"),
        ("What are the company OKRs?",               "mission",       "OKR query"),
    ]

    print("\n  Testing semantic retrieval accuracy...")
    for query, expected_keyword, label in retrieval_tests:
        result = retrieve_docs(query, n_results=3)
        check(
            label + " returns relevant content",
            expected_keyword.lower() in result.lower(),
            f"keyword '{expected_keyword}' not found in result"
        )

    # retrieve_docs_for_role works for each role
    print("\n  Testing role-filtered retrieval...")
    role_tests = [
        ("How do I do a code review?",       "engineering", "engineering"),
        ("How do I build my sales pipeline?","sales",       "sales"),
        ("How do I handle churn risk?",      "customer_success", "customer_success"),
    ]
    for query, role, label in role_tests:
        result = retrieve_docs_for_role(query, role, n_results=3)
        check(
            f"Role-filtered retrieval works for '{label}'",
            bool(result) and len(result) > 50,
            f"got {len(result)} chars"
        )

    # get_role_from_persona mapping
    print("\n  Testing persona to role mapping...")
    persona_tests = [
        ("new_hire", "all"),
        ("manager",  "sales"),
        ("ops",      "all"),
        ("exec",     "customer_success"),
    ]
    for persona, expected_role in persona_tests:
        result = get_role_from_persona(persona)
        check(
            f"Persona '{persona}' maps to role '{expected_role}'",
            result == expected_role,
            f"got '{result}'"
        )

    # Edge cases
    print("\n  Testing edge cases...")
    empty_result = retrieve_docs("xyzzy nonsense query that matches nothing relevant")
    check("Edge case: nonsense query returns string",  isinstance(empty_result, str))
    check("Edge case: nonsense query does not crash",  True)

    unknown_role = retrieve_docs_for_role("What is the PTO policy?", "unknown_role")
    check("Edge case: unknown role falls back gracefully", bool(unknown_role))

    print("="*55)
    print(f"  RESULTS: {passed} passed, {failed} failed")
    if failed == 0:
        print("  🎉 All RAG tests passed!")
    else:
        print("  ⚠️  Some tests failed — check output above")
    print("="*55 + "\n")

    return failed == 0


# ─── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "="*55)
    print("  VELO — RAG Module")
    print("  enterprise-knowledge-agent")
    print("="*55)

    if not validate_config():
        sys.exit(1)

    success = run_tests()
    sys.exit(0 if success else 1)
