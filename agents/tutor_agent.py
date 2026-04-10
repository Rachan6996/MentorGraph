import os
from groq import Groq

client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def tutor_agent(question: str, context: list, summary: str = "", recent_turns: str = ""):
    """
    Uses RAG context plus rolling memory summary to explain + ask follow-up.
    Strictly answers from context only.
    """

    if not context and not summary and not recent_turns:
        return '{"explanation": "I don\'t have any documents loaded on this topic. Please upload a relevant document in the Discovery Hub first.", "follow_up_question": "Which subject would you like to upload study material for?"}'

    context_text = "\n\n".join(context) if context else ""

    # Build memory block
    memory_block = ""
    if summary:
        memory_block += f"\nSession Summary (compressed memory):\n{summary}\n"
    if recent_turns:
        memory_block += f"\nRecent Conversation (use this for follow-up questions):\n{recent_turns}\n"

    # If no context but we have session memory, use memory as the source
    if not context_text and (summary or recent_turns):
        context_section = f"(No new documents matched. Answer based on the session memory below.)"
    else:
        context_section = f"Knowledge Base Context:\n{context_text}"

    prompt = f"""You are a strict AI Tutor with memory of the ongoing session.
You MUST answer using the provided information below.
For follow-up questions, use the Recent Conversation and Session Summary as your primary source.
Do NOT use any outside knowledge. Do NOT make up information.
If neither the context nor the memory contains enough information, say:
"I don't have enough information on this topic in my knowledge base."
{memory_block}
{context_section}

Student Question:
{question}

Examples:
1. Question: "What is evaporation?"
   Context: "Evaporation is the process where liquid water turns into water vapor when heated."
   Explanation: "Evaporation is when liquid water turns into vapor because of heat."
   Follow-up: "Can you name one source of heat that causes this?"

2. Question: "Why are plants green?"
   Context: "Plants contain chlorophyll, a green pigment that absorbs sunlight for photosynthesis."
   Explanation: "Plants are green because of a pigment called chlorophyll, which absorbs sunlight."
   Follow-up: "What does the plant do with that absorbed sunlight?"

Return ONLY JSON (no extra text):
{{
  "explanation": "clear explanation using ONLY the context above",
  "follow_up_question": "one question to test understanding of the above explanation"
}}
"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )

    return response.choices[0].message.content