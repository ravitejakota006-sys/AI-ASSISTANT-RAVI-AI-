import os, json, math, uuid
from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from pypdf import PdfReader
from google import genai
from google.genai import types
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

load_dotenv()
BASE = Path(__file__).resolve().parent.parent
STATIC = BASE / "static"
MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
EMBED = os.getenv("EMBEDDING_MODEL", "gemini-embedding-001")
COLLECTION = os.getenv("QDRANT_COLLECTION", "ravi_ai_docs_gemini")

app = FastAPI(title="RaviAI – Agentic Career Assistant", version="2.0.0")
app.mount("/static", StaticFiles(directory=STATIC), name="static")

GEMINI_KEY = os.getenv("GEMINI_API_KEY")
gemini = genai.Client(api_key=GEMINI_KEY) if GEMINI_KEY else None
qdrant = QdrantClient(url=os.getenv("QDRANT_URL", "http://localhost:6333"))

class ChatRequest(BaseModel):
    message: str
    history: list[dict[str, str]] = []

class MatchRequest(BaseModel):
    resume: str
    job_description: str

def require_gemini():
    if not gemini:
        raise HTTPException(500, "Set GEMINI_API_KEY in .env first")

def ensure_qdrant():
    try:
        names = [c.name for c in qdrant.get_collections().collections]
        if COLLECTION not in names:
            qdrant.create_collection(
                COLLECTION,
                vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
            )
        return True
    except Exception:
        return False

def embed(text: str):
    require_gemini()
    result = gemini.models.embed_content(
        model=EMBED,
        contents=text,
        config=types.EmbedContentConfig(
            output_dimensionality=1536,
            task_type="RETRIEVAL_DOCUMENT",
        ),
    )
    return result.embeddings[0].values

def embed_query(text: str):
    require_gemini()
    result = gemini.models.embed_content(
        model=EMBED,
        contents=text,
        config=types.EmbedContentConfig(
            output_dimensionality=1536,
            task_type="RETRIEVAL_QUERY",
        ),
    )
    return result.embeddings[0].values

def chunk_text(text: str, size=900, overlap=120):
    text = " ".join(text.split())
    chunks=[]; start=0
    while start < len(text):
        end=min(len(text), start+size)
        chunks.append(text[start:end])
        if end == len(text): break
        start=end-overlap
    return chunks

def search_docs(query: str, limit=5):
    require_gemini()
    if not ensure_qdrant(): return []
    try:
        hits=qdrant.search(
            collection_name=COLLECTION,
            query_vector=embed_query(query),
            limit=limit,
        )
        return [h.payload.get("text", "") for h in hits if h.payload]
    except Exception:
        return []

def calculator(expression: str) -> str:
    """Calculate a basic arithmetic expression using numbers and +, -, *, /, %, and parentheses."""
    allowed=set("0123456789+-*/().% ")
    if not expression or any(c not in allowed for c in expression):
        return "Invalid expression"
    try:
        return str(eval(expression, {"__builtins__": {}}, {}))
    except Exception:
        return "Could not calculate"

def tool_search_documents(query: str) -> str:
    """Search uploaded PDFs and documents using semantic vector search.

    Use this when the user asks about information contained in an uploaded document.
    """
    docs=search_docs(query)
    return "\n\n---\n\n".join(docs) if docs else "No uploaded document content matched this query."

SYSTEM="""You are RaviAI, a practical agentic career and productivity assistant.
Be concise but useful. You can use tools when needed.
For questions about uploaded documents, use search_documents instead of inventing facts.
Explain when an answer is based on retrieved document context.
For calculations, use the calculator tool.
Help users understand AI, programming, resumes and job preparation.
Never claim a tool was used if it was not."""

def make_contents(message, history):
    contents=[]
    for m in history[-8:]:
        if m.get("role") in ("user", "model", "assistant"):
            role = "model" if m.get("role") == "assistant" else m.get("role")
            contents.append(types.Content(role=role, parts=[types.Part.from_text(text=m.get("content", ""))]))
    contents.append(types.Content(role="user", parts=[types.Part.from_text(text=message)]))
    return contents

def run_agent(message, history):
    require_gemini()
    response = gemini.models.generate_content(
        model=MODEL,
        contents=make_contents(message, history),
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM,
            tools=[calculator, tool_search_documents],
            temperature=0.2,
        ),
    )
    return response.text or "I couldn't produce an answer."

@app.get("/")
def home():
    return FileResponse(STATIC/"index.html")

@app.post("/api/chat")
def chat(req: ChatRequest):
    return {"answer": run_agent(req.message, req.history)}

@app.post("/api/upload")
async def upload(file: UploadFile=File(...)):
    require_gemini()
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400,"PDF only for this demo")
    raw=await file.read()
    tmp=BASE/"data"/f"{uuid.uuid4()}.pdf"
    tmp.write_bytes(raw)
    try:
        text="\n".join((p.extract_text() or "") for p in PdfReader(str(tmp)).pages)
        chunks=chunk_text(text)
        ensure_qdrant()
        points=[]
        for i,c in enumerate(chunks):
            points.append(PointStruct(
                id=str(uuid.uuid4()),
                vector=embed(c),
                payload={"text":c,"source":file.filename,"chunk":i},
            ))
        qdrant.upsert(collection_name=COLLECTION,points=points)
        return {"message":f"Indexed {len(chunks)} chunks from {file.filename}","chunks":len(chunks)}
    finally:
        tmp.unlink(missing_ok=True)

@app.post("/api/match")
def match(req: MatchRequest):
    require_gemini()
    prompt=f"""Compare this RESUME and JOB DESCRIPTION. Return ONLY valid JSON with these keys:
match_score (0-100), matching_skills (array), missing_skills (array), strengths (array), recommendations (array), interview_topics (array).
Do not invent skills not supported by the resume.

RESUME:\n{req.resume}\n\nJOB DESCRIPTION:\n{req.job_description}"""
    response = gemini.models.generate_content(
        model=MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction="You are an ATS and career analyst. Return valid JSON only.",
            response_mime_type="application/json",
            temperature=0.1,
        ),
    )
    try:
        return json.loads(response.text)
    except Exception:
        raise HTTPException(502, "Gemini returned an invalid JSON response")
