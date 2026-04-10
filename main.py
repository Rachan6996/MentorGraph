import os
import json
import faiss
from fastapi import FastAPI, APIRouter, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

# Internal Imports
from rag.vector_store import search, add_documents
from rag.load_data import load_documents_from_folder
from agents.workflow import graph, evaluator_node
from agents.evaluator_agent import batch_evaluator_agent, question_generator
from utils.db import save_decision, get_history
from utils.memory import (
    add_to_short_term,
    get_recent_turns,
    get_summary,
    update_summary,
    load_summary_from_disk
)
from utils.quiz_session import add_answer, get_session, get_count, clear_session


app = FastAPI()

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Models ---
class AskRequest(BaseModel):
    question: str

class AnswerRequest(BaseModel):
    question: str
    student_answer: str

class QuizAnswerRequest(BaseModel):
    question: str
    student_answer: str

class QuizReviewRequest(BaseModel):
    context_query: str = ""

class ReviewRequest(BaseModel):
    original_ai_evaluation: dict
    human_score: int
    human_feedback: str

# --- Startup ---
@app.on_event("startup")
def startup_event():
    if os.path.exists("data"):
        load_documents_from_folder("data")
    load_summary_from_disk()  # Restore compressed memory from last session

@app.get("/")
def root():
    return FileResponse("index.html")

# --- Routes ---

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    os.makedirs("data", exist_ok=True)
    file_path = os.path.join("data", file.filename)
    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())

    from rag.vector_store import reset_knowledge_base
    reset_knowledge_base()
    load_documents_from_folder("data")

    return {"status": "success", "message": f"File {file.filename} uploaded and indexed successfully."}

@app.post("/ask")
def ask_question(req: AskRequest):
    context = search(req.question)

    # Inject memory into graph state
    initial_state = {
        "question": req.question,
        "context": context,
        "tutor_explanation": None,
        "follow_up": None,
        "student_answer": None,
        "score": None,
        "feedback": None,
        "summary": get_summary(),
        "recent_turns": get_recent_turns()
    }

    result = graph.invoke(initial_state)

    explanation = result["tutor_explanation"] or ""
    follow_up = result["follow_up"] or ""

    # Update rolling memory (async-safe: runs in same thread)
    add_to_short_term(req.question, explanation)
    update_summary(req.question, explanation)

    # Save to history as 'ask' type (chat history)
    save_decision(
        question=req.question,
        ai_answer={"explanation": explanation, "follow_up": follow_up},
        ai_evaluation={"score": 0, "feedback": "Chat session record"}, # Scores usually N/A for chat
        sources=context,
        record_type="ask"
    )

    return {
        "status": "success",
        "tutor_response": {
            "explanation": explanation,
            "follow_up_question": follow_up
        },
        "sources": context
    }

@app.post("/answer")
def submit_answer(req: AnswerRequest):
    context = search(req.question)

    state = {
        "question": req.question,
        "student_answer": req.student_answer,
        "context": context,
        "score": None,
        "feedback": None
    }

    result = evaluator_node(state)
    ai_evaluation = {"score": result["score"], "feedback": result["feedback"]}

    # Save evaluation decision (lightweight — only scores, not full chats)
    save_decision(
        question=req.question,
        ai_answer={"explanation": "", "follow_up": ""},
        ai_evaluation=ai_evaluation,
        sources=context,
        record_type="ask"
    )

    return {
        "question": req.question,
        "student_answer": req.student_answer,
        "evaluation": ai_evaluation,
        "sources": context
    }

@app.post("/review")
def review_evaluation(req: ReviewRequest):
    human_decision = {"score": req.human_score, "feedback": req.human_feedback}

    history = get_history()
    if history:
        latest = history[-1]
        save_decision(
            question=latest["question"],
            ai_answer={"explanation": latest.get("ai_answer", ""), "follow_up": ""},
            ai_evaluation={"score": latest["score"], "feedback": latest["feedback"]},
            sources=latest.get("sources", []),
            human_override=human_decision
        )

    return {
        "status": "stored",
        "message": "Human review captured and final score updated."
    }

@app.get("/history")
def show_history():
    return get_history()

@app.get("/final-output")
def final_output():
    """Returns the unified Final Output JSON matching the assessment spec."""
    history = get_history()
    output = []
    for record in history:
        output.append({
            "question": record.get("question", ""),
            "ai_answer": record.get("ai_answer", ""),
            "ai_score": record.get("score", 0),
            "final_score": record.get("final_score", record.get("score", 0)),
            "feedback": record.get("feedback", ""),
            "sources": record.get("sources", []),
            "type": record.get("type", "ask")
        })
    return output

@app.get("/memory")
def show_memory():
    """Debug endpoint to inspect the current rolling summary."""
    return {
        "summary": get_summary(),
        "recent_turns": get_recent_turns()
    }

# ── QUIZ ENDPOINTS ─────────────────────────────────────────────────────────────

@app.post("/quiz/next")
def quiz_next(req: QuizAnswerRequest):
    """Save current Q&A, return a freshly generated next question."""
    add_answer(req.question, req.student_answer)
    context = search(req.question, top_k=3)
    next_q = question_generator(context)
    return {"status": "saved", "answered_count": get_count(), "next_question": next_q}

@app.post("/quiz/review")
def quiz_review(req: QuizReviewRequest):
    """Batch-evaluate all answered questions. Save overall report to history."""
    import json as _json
    session = get_session()
    if not session:
        return {"status": "empty", "message": "No answers recorded yet."}
    query = req.context_query or session[-1]["question"]
    context = search(query, top_k=5)
    # batch_evaluator_agent now returns a parsed dict directly
    report = batch_evaluator_agent(session, context)
    save_decision(
        question=f"Quiz Session ({len(session)} questions)",
        ai_answer={"explanation": _json.dumps(report.get("results", [])), "follow_up": ""},
        ai_evaluation={"score": report.get("overall_score", 0), "feedback": report.get("overall_feedback", "")},
        sources=context,
        record_type="quiz"
    )
    return {"status": "reviewed", "session_count": len(session), "report": report}

@app.post("/quiz/reset")
def quiz_reset():
    """Clear the in-RAM quiz session."""
    clear_session()
    return {"status": "reset", "message": "Quiz session cleared."}

@app.get("/quiz/status")
def quiz_status():
    return {"answered_count": get_count(), "session": get_session()}
