import os
from typing import List, Dict, Any, Optional

import uvicorn
import chromadb
from chromadb.config import Settings
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import SentenceTransformer
import requests

# =========================
# CONFIG
# =========================
CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "notion_links"  # must match ingestion
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
OLLAMA_API = os.getenv("OLLAMA_API", "http://localhost:11434/api/generate")

# =========================
# INIT (lazy + defensive)
# =========================
app = FastAPI(title="iCEP Assistant")

# Allow local file:// index_notion.html to hit http://localhost:8000
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten later if you want
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_embedder: Optional[SentenceTransformer] = None
_chroma_client: Optional[chromadb.ClientAPI] = None
_collection = None

def get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBED_MODEL_NAME)
    return _embedder

def get_chroma():
    global _chroma_client
    if _chroma_client is None:
        _chroma_client = chromadb.PersistentClient(
            path=CHROMA_DIR,
            settings=Settings(anonymized_telemetry=False)
        )
    return _chroma_client

def get_collection():
    global _collection
    if _collection is None:
        client = get_chroma()
        # Try to get; if missing, don’t crash—return None and we’ll 503 on /ask
        try:
            _collection = client.get_collection(name=COLLECTION_NAME)
        except Exception:
            _collection = None
    return _collection

# =========================
# MODELS
# =========================
class AskRequest(BaseModel):
    question: str
    top_k: int = 3
    model: str = "llama3:8b"

class SourceItem(BaseModel):
    preview: str
    chunk_id: str
    similarity: float

class AskResponse(BaseModel):
    answer: str
    sources: List[SourceItem]

# =========================
# HELPERS
# =========================
def embed_query(text: str) -> List[float]:
    return get_embedder().encode([text])[0].tolist()

def similarity_from_distance(distance: float) -> float:
    sim = 1.0 - float(distance)
    return max(0.0, min(1.0, sim))

def call_ollama(model_name: str, prompt: str) -> str:
    payload = {"model": model_name, "prompt": prompt, "stream": False}
    r = requests.post(OLLAMA_API, json=payload, timeout=120)
    r.raise_for_status()
    data = r.json()
    return (data.get("response") or "").strip()

def build_rag_prompt(question: str, docs: List[Dict[str, str]]) -> str:
    ctx_lines = []
    for i, d in enumerate(docs, start=1):
        ctx_lines.append(f"[Source {i}] {d['title']}")
        if d.get("url"):
            ctx_lines.append(f"Link: {d['url']}")
        ctx_lines.append(f"Content:\n{d['text']}\n")
    context = "\n".join(ctx_lines)
    return (
        "You are a helpful assistant. Use only the sources below. "
        "If the user wants links, list the most relevant URLs clearly.\n\n"
        f"QUESTION:\n{question}\n\nSOURCES:\n{context}\n\n"
        "Answer the question. If unsure, say you don't know."
    )

def ollama_available() -> bool:
    try:
        # Cheap ping: the generate endpoint with tiny prompt
        requests.post(OLLAMA_API, json={"model": "llama3:8b", "prompt": "ping", "stream": False}, timeout=3)
        return True
    except Exception:
        return False

# =========================
# DEBUG ROUTES
# =========================
@app.get("/health")
def health():
    col = get_collection()
    return {
        "ok": True,
        "collection_found": col is not None,
        "ollama_available": ollama_available(),
    }

@app.get("/collections")
def collections():
    client = get_chroma()
    try:
        cols = client.list_collections()
        return {"collections": [c.name for c in cols]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chroma error: {e}")

# =========================
# MAIN ROUTE
# =========================
@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    col = get_collection()
    if col is None:
        raise HTTPException(
            status_code=503,
            detail=f"Chroma collection '{COLLECTION_NAME}' not found. "
                   f"Run ingestion first, or verify COLLECTION_NAME matches ingestion."
        )

    # 1) Embed + retrieve
    qvec = embed_query(req.question)
    try:
        res = col.query(
            query_embeddings=[qvec],
            n_results=req.top_k,
            include=["documents", "metadatas", "distances"],  # 'ids' returned implicitly
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chroma query failed: {e}")

    ids   = res.get("ids", [[]])[0]
    docs  = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0]

    if not ids:
        return AskResponse(answer="I couldn't find anything for that.", sources=[])

    # 2) RAG docs + sources
    rag_docs: List[Dict[str, Any]] = []
    sources: List[SourceItem] = []
    for i in range(len(ids)):
        meta  = metas[i] if i < len(metas) else {}
        url   = meta.get("url")
        title = meta.get("title") or "(Untitled)"
        text  = docs[i]
        sim   = similarity_from_distance(dists[i] if i < len(dists) else 1.0)

        rag_docs.append({"title": title, "url": url, "text": text})
        sources.append(SourceItem(preview=(url or title), chunk_id=ids[i], similarity=sim))

    # 3) Pick ONLY the single most relevant link
    top_link = None
    for s in sources:
        if isinstance(s.preview, str) and s.preview.startswith(("http://", "https://")):
            top_link = s.preview
            break

    # 4) Try LLM — if it fails/absent, stay silent (no banner)
    answer = ""
    try:
        if ollama_available():
            prompt = build_rag_prompt(req.question, rag_docs)
            answer = call_ollama(req.model, prompt) or ""
    except Exception:
        # swallow errors to keep UX clean; we still show the link below
        pass

    # 5) Always append only the top link if present
    if top_link:
        answer = (answer + "\n\n" if answer else "") + f"Most relevant link:\n{top_link}"
    else:
        answer = answer or "(No link found for this query.)"

    return AskResponse(answer=answer, sources=sources)



if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
