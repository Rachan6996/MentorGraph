# MentorGraph 

MentorGraph is an advanced multi-agent RAG system designed to orchestrate educational tutoring and automated evaluation with precision. It integrates a seamless Human-in-the-Loop workflow to ensure every AI decision is transparent, reviewable, and expert-aligned.

## Core Features
- **Intelligent RAG**: Context-aware retrieval using FAISS and Hybrid Search (BM25).
- **Multi-Agent Workflow**: Stateful orchestration via **LangGraph** for Tutor and Evaluator agents.
- **Human-in-the-Loop**: Professional Admin panel for overriding AI scores and providing sentiment-tagged feedback.
- **Unified Learning History**: Comprehensive logs of chat sessions, quiz results, and human reviews.
- **Fast & Modern UI**: Sleek glassmorphic interface with real-time feedback.

## Tech Stack
- **Backend**: FastAPI (Python)
- **Orchestration**: LangGraph
- **LLM**: Groq (Llama 3.3 70B)
- **Vector DB**: FAISS
- **Frontend**: Vanilla JS / CSS3

## ⚡ Quick Start

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Set Environment Variables**:
   Create a `.env` file with:
   ```env
   GROQ_API_KEY=your_api_key_here
   ```

3. **Run the Server**:
   ```bash
   uvicorn main:app --reload
   ```

## Project Structure
- `agents/`: Tutor and Evaluator logic + LangGraph workflow.
- `rag/`: Document loading, chunking, and vector storage.
- `utils/`: Persistent history and session management.
- `data/`: Knowledge base source documents.

---
*Created for the AI Engineer Assessment (Education Domain).*
