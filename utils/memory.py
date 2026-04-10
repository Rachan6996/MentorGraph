"""
utils/memory.py

Rolling Summary Memory System:
- Keeps last 2 turns in RAM (short-term memory)
- Generates/updates a compressed summary after each turn
- Persists ONLY the summary to disk (overwrites, never appends)
- Importance filter: skips trivial/redundant queries
"""

import os
import json
from groq import Groq

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

MEMORY_FILE = "session_summary.json"

# --- In-RAM short-term memory (last 2 turns only) ---
_short_term: list[dict] = []  # [{"role": "user"|"ai", "content": "..."}]
_rolling_summary: str = ""     # Compressed session summary


# ── Importance Filter ──────────────────────────────────────────────────────────
TRIVIAL_PATTERNS = [
    "ok", "okay", "thanks", "thank you", "got it", "sure",
    "yes", "no", "hi", "hello", "bye", "good", "great", "nice"
]

def is_important(query: str) -> bool:
    """
    Returns False if the query is trivially small talk or redundant.
    """
    normalized = query.strip().lower().rstrip("!?.")
    if normalized in TRIVIAL_PATTERNS:
        return False
    if len(normalized.split()) <= 2:
        return False
    return True


# ── Short-Term Memory (RAM) ────────────────────────────────────────────────────
def add_to_short_term(user_query: str, ai_response: str):
    """
    Stores latest turn in RAM. Keeps only last 2 turns.
    """
    global _short_term
    _short_term.append({"role": "user", "content": user_query})
    _short_term.append({"role": "ai", "content": ai_response})
    # Keep only last 4 messages (2 turns)
    _short_term = _short_term[-4:]


def get_recent_turns() -> str:
    """
    Returns the last 1-2 turns as a formatted string for prompt injection.
    """
    if not _short_term:
        return ""
    lines = []
    for msg in _short_term:
        prefix = "Student" if msg["role"] == "user" else "Tutor"
        lines.append(f"{prefix}: {msg['content']}")
    return "\n".join(lines)


# ── Rolling Summary ────────────────────────────────────────────────────────────
def get_summary() -> str:
    return _rolling_summary


def update_summary(user_query: str, ai_response: str):
    """
    Generates a concise updated summary of the conversation.
    Only runs if the query passes the importance filter.
    Persists to disk (overwrites).
    """
    global _rolling_summary

    if not is_important(user_query):
        return  # Skip small talk

    prev_summary = _rolling_summary or "No prior context."

    prompt = f"""You are a memory compression assistant. Your job is to update a concise session summary.

Previous Summary:
{prev_summary}

Latest Student Query:
{user_query}

Latest Tutor Response:
{ai_response}

Rules:
- Focus on: user intent, key concepts covered, any decisions or constraints introduced
- Ignore small talk or redundant repetitions
- Keep the summary under 100 words
- Write in third person (e.g., "Student asked about...")

Return ONLY the updated summary text, no extra formatting."""

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=200
        )
        _rolling_summary = response.choices[0].message.content.strip()
        _persist_summary()
    except Exception:
        pass  # Don't crash the app if summary fails


def _persist_summary():
    """
    Overwrites the summary file — never appends.
    """
    with open(MEMORY_FILE, "w") as f:
        json.dump({"summary": _rolling_summary}, f, indent=2)


def load_summary_from_disk():
    """
    Loads the last persisted summary on server startup.
    """
    global _rolling_summary
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, "r") as f:
            data = json.load(f)
            _rolling_summary = data.get("summary", "")
