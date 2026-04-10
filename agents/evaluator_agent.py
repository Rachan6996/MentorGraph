import os
import json
import re
from groq import Groq

from utils.json_helper import extract_json

client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def evaluator_agent(question: str, student_answer: str, context: list):
    """Single Q&A evaluation (kept for compatibility)."""
    context_text = "\n".join(context)
    prompt = f"""You are an evaluator. Grade the student's answer based on the context.

Context:
{context_text}

Question:
{question}

Student Answer:
{student_answer}

Examples:
1. Student Answer: "It moves water."
   Score: 4
   Feedback: "Correct but very basic. Mention evaporation or condensation."

2. Student Answer: "Photosynthesis uses sunlight and CO2 to make glucose and oxygen."
   Score: 10
   Feedback: "Perfect! You covered both the inputs and the byproducts."

Return ONLY JSON:
{{
  "score": 0-10,
  "feedback": "short feedback"
}}
"""
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )
    return response.choices[0].message.content


def batch_evaluator_agent(qa_pairs: list, context: list) -> dict:
    """
    Evaluates a full quiz session (multiple Q&A pairs) at once.
    Returns a parsed dict with per-question results + overall score.
    """
    context_text = "\n".join(context) if context else "No specific context provided."

    qa_block = ""
    for i, pair in enumerate(qa_pairs, start=1):
        qa_block += f"\nQ{i}: {pair['question']}\nStudent Answer: {pair['student_answer']}\n"

    prompt = f"""You are a thorough AI Evaluator reviewing a complete quiz session.

Knowledge Base Context:
{context_text}

Quiz Answers:
{qa_block}

For each question, provide:
- A score from 0 to 10
- Short, constructive feedback

Then provide an overall_score (numeric average) and overall_feedback (1-2 sentence summary).

Return ONLY valid JSON, no markdown, no extra text:
{{
  "results": [
    {{"question": "...", "student_answer": "...", "score": 7, "feedback": "..."}},
    ...
  ],
  "overall_score": 7,
  "overall_feedback": "..."
}}
"""
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )
    raw = response.choices[0].message.content
    parsed = extract_json(raw)
    if parsed:
        return parsed
    return {"results": [], "overall_score": 0, "overall_feedback": "Could not parse evaluation. Raw: " + raw[:200]}


def question_generator(topic_context: list) -> str:
    """
    Generates a new quiz question based on the loaded knowledge base context.
    """
    context_text = "\n".join(topic_context) if topic_context else ""
    if not context_text:
        return "What is the main concept you have learned so far?"

    prompt = f"""Based on the following educational content, generate ONE concise quiz question to test student understanding.

Content:
{context_text}

Rules:
- Ask about a specific fact or concept from the content
- Keep it clear and unambiguous
- Do not repeat obvious or trivial questions

Return ONLY the question text, no extra formatting."""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=80
    )
    return response.choices[0].message.content.strip()