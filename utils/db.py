import json
import os

DB_FILE = "storage.json"

def save_decision(
    question: str,
    ai_answer: dict, # contains explanation + follow_up
    ai_evaluation: dict,
    sources: list = None,
    human_override: dict = None,
    record_type: str = "ask" # "ask" or "quiz"
):
    """
    Stores session state with a record type to separate chat vs quiz history
    """
    data = []
    if os.path.exists(DB_FILE):
        with open(DB_FILE, "r") as f:
            try:
                data = json.load(f)
            except:
                data = []
            
    record = {
        "type": record_type,
        "question": question,
        "ai_answer": f"{ai_answer.get('explanation') if ai_answer else ''}\n\nFollow-up: {ai_answer.get('follow_up') if ai_answer else ''}",
        "score": ai_evaluation.get("score") if ai_evaluation else 0,
        "final_score": human_override.get("score") if human_override else (ai_evaluation.get("score") if ai_evaluation else 0),
        "feedback": human_override.get("feedback") if human_override else (ai_evaluation.get("feedback") if ai_evaluation else ""),
        "sources": sources or []
    }
    data.append(record)
    
    with open(DB_FILE, "w") as f:
        json.dump(data, f, indent=2)

def get_history():
    if os.path.exists(DB_FILE):
        with open(DB_FILE, "r") as f:
            try:
                return json.load(f)
            except:
                return []
    return []
