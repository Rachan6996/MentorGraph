"""
utils/quiz_session.py

In-RAM quiz session storage.
Accumulates Q&A pairs during an active quiz.
No scoring per question — review happens on demand.
"""

_quiz_session: list[dict] = []  # [{"question": "...", "student_answer": "..."}]


def add_answer(question: str, student_answer: str):
    """Add a new Q&A pair to the current session."""
    _quiz_session.append({
        "question": question,
        "student_answer": student_answer
    })


def get_session() -> list[dict]:
    """Return all answered questions so far."""
    return list(_quiz_session)


def get_count() -> int:
    return len(_quiz_session)


def clear_session():
    """Reset the quiz session."""
    global _quiz_session
    _quiz_session = []
