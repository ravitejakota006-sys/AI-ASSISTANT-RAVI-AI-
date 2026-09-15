# RaviAI – Agentic Career & Productivity Assistant

A portfolio-ready AI assistant aligned with the Kloudbricks AI Developer / Junior AI Engineer JD.

## Features
- LLM-powered conversational assistant
- Function/tool calling
- Calculator tool
- PDF ingestion and semantic RAG
- OpenAI embeddings + Qdrant vector database
- Resume/JD match analysis endpoint with structured JSON
- Conversation context
- Clean responsive web UI
- Docker Compose for Qdrant

## Stack
Python, FastAPI, OpenAI API, Qdrant, embeddings, REST APIs, HTML/CSS/JavaScript, Docker.

## Run
1. Install Python 3.10+
2. `python -m venv .venv`
3. macOS/Linux: `source .venv/bin/activate` | Windows: `.venv\\Scripts\\activate`
4. `pip install -r requirements.txt`
5. Copy `.env.example` to `.env` and add `OPENAI_API_KEY`.
6. Start Qdrant: `docker compose up -d`
7. Start app: `uvicorn app.main:app --reload`
8. Open `http://127.0.0.1:8000`

## Demo flow
1. Upload your resume PDF.
2. Ask: “Summarize my resume and list my strongest skills.”
3. Upload the Kloudbricks JD PDF.
4. Ask: “What are the top missing skills I should learn for this role?”
5. Try a calculation to demonstrate tool calling.

## Resume bullet
**RaviAI – Agentic Career Assistant | Python, FastAPI, OpenAI, RAG, Qdrant, REST APIs, Docker**
Built an LLM-powered career assistant with semantic document search, PDF ingestion, embeddings, Qdrant vector storage, function calling, contextual conversations, and structured resume–JD analysis.

## Next upgrades
- LangGraph agent orchestration
- PostgreSQL persistent memory
- n8n workflow automation
- Multimodal image analysis
- Authentication
- Dockerize the FastAPI app and deploy to AWS
