import os
import json
from groq import Groq

client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def tutor_agent(question: str, context: list, summary: str = "", recent_turns: str = ""):
    """
    Uses RAG context plus rolling memory summary to explain + ask follow-up.
    The LLM handles intent detection naturally — no hardcoded logic.
    """

    context_text = "\n\n".join(context) if context else ""

    # Build memory block
    memory_block = ""
    if summary:
        memory_block += f"\nSession Summary (compressed memory):\n{summary}\n"
    if recent_turns:
        memory_block += f"\nRecent Conversation (use this for follow-up questions):\n{recent_turns}\n"

    # If no context but we have session memory, use memory as the source
    if not context_text and (summary or recent_turns):
        context_section = "(No new documents matched. Answer based on the session memory below.)"
    elif context_text:
        context_section = f"Knowledge Base Context:\n{context_text}"
    else:
        context_section = "(No documents loaded yet.)"

    prompt = f"""You are a world-class AI Tutor — friendly, clear, and thorough, just like ChatGPT.

STEP 1 — DETECT INTENT:
Read the student's message carefully. Determine if it is:
A) A greeting or small talk (e.g. "Hi", "Hello", "kya kr rhe ho", "sup")
B) A knowledge question about a topic

STEP 2 — RESPOND ACCORDINGLY:

If (A) Greeting/Small talk:
- Respond warmly and conversationally, like a friendly teacher would.
- Ask what topic they'd like to study today.
- Do NOT include any technical content or reference session memory.
- Keep it short and natural.

If (B) Knowledge question:
- Start with a **one-line bold summary** of the answer.
- Then explain using a **Simple way to understand it** section with bullet points.
- Add a **Real-life example** to make it relatable.
- End with **Key takeaways** as a numbered list (2-3 points max).
- Use markdown: **bold**, bullet points, numbered lists.
- If the context covers the topic, use it. If not, use general knowledge and note "(Based on general knowledge)" at the end.

{memory_block}
{context_section}

Student Message:
{question}

Return ONLY valid JSON (no extra text before or after):
{{
  "explanation": "your response here",
  "follow_up_question": "one engaging question to continue the conversation"
}}
"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=1024
    )

    return response.choices[0].message.content