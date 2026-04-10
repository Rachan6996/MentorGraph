import json
from typing import TypedDict, List, Optional
from langgraph.graph import StateGraph, END
from agents.tutor_agent import tutor_agent
from agents.evaluator_agent import evaluator_agent
from utils.json_helper import extract_json

# 1. Define State
class AgentState(TypedDict):
    question: str
    context: List[str]
    tutor_explanation: Optional[str]
    follow_up: Optional[str]
    student_answer: Optional[str]
    score: Optional[int]
    feedback: Optional[str]
    summary: Optional[str]
    recent_turns: Optional[str]

# 2. Define Nodes
def tutor_node(state: AgentState):
    res_raw = tutor_agent(
        state["question"],
        state["context"],
        summary=state.get("summary", ""),
        recent_turns=state.get("recent_turns", "")
    )
    data = extract_json(res_raw)
    
    if not data:
        return {
            "tutor_explanation": res_raw,
            "follow_up": "Do you have any thoughts on this?"
        }
    
    return {
        "tutor_explanation": data.get("explanation"),
        "follow_up": data.get("follow_up_question")
    }

def evaluator_node(state: AgentState):
    res_raw = evaluator_agent(state["question"], state["student_answer"], state["context"])
    data = extract_json(res_raw)
        
    if not data:
        return {
            "score": 0,
            "feedback": f"Could not parse AI evaluation. Raw: {res_raw}"
        }

    return {
        "score": data.get("score", 0),
        "feedback": data.get("feedback", "No feedback provided.")
    }

# 3. Build Graph
workflow = StateGraph(AgentState)

workflow.add_node("tutor_agent", tutor_node)
workflow.add_node("evaluator_agent", evaluator_node)

workflow.set_entry_point("tutor_agent")
workflow.add_edge("tutor_agent", END)

# We can manually trigger nodes in the API
graph = workflow.compile()
