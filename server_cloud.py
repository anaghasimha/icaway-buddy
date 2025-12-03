import os
from typing import List, Dict, Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import requests
from dotenv import load_dotenv

load_dotenv()

# ========= CONFIG =========
EXCEL_PATH = "icaway_lms_links.xlsx"   # your Excel file with Name + Link columns
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "llama-3.3-70b-versatile"   # or "llama3-8b-8192"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
# ==========================

if not GROQ_API_KEY:
    print("WARNING: GROQ_API_KEY is not set. /ask will fail until you configure it.")

app = FastAPI(title="ICAway Buddy Cloud")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # you can restrict this later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========= DATA & EMBEDDINGS =========
class LinkItem(BaseModel):
    id: str
    name: str
    url: str

links_df: pd.DataFrame | None = None
link_embeddings: np.ndarray | None = None
embedder: SentenceTransformer | None = None


def load_links_and_embeddings():
    global links_df, link_embeddings, embedder

    if not os.path.exists(EXCEL_PATH):
        raise RuntimeError(f"Excel file '{EXCEL_PATH}' not found in working directory.")

    df = pd.read_excel(EXCEL_PATH)
    if "Name" not in df.columns or "Link" not in df.columns:
        raise RuntimeError("Excel must have columns: 'Name' and 'Link'.")

    df = df.copy()
    df["Name"] = df["Name"].astype(str).str.strip()
    df["Link"] = df["Link"].astype(str).str.strip()
    df = df[df["Link"].str.startswith("http")].reset_index(drop=True)

    if df.empty:
        raise RuntimeError("No valid links found in Excel.")

    embedder = SentenceTransformer(EMBED_MODEL_NAME)
    texts = (df["Name"] + "\n" + df["Link"]).tolist()
    emb = embedder.encode(texts, normalize_embeddings=True)
    link_embeddings = np.array(emb, dtype=np.float32)
    df["id"] = df.index.astype(str)

    links_df = df
    print(f"Loaded {len(df)} links from Excel and created embeddings.")


def cosine_top_k(question: str, k: int = 3) -> List[LinkItem]:
    if links_df is None or link_embeddings is None or embedder is None:
        raise RuntimeError("Embeddings not initialized.")

    q_emb = embedder.encode([question], normalize_embeddings=True)[0]
    sims = np.dot(link_embeddings, q_emb)
    top_idx = np.argsort(-sims)[:k]

    items: List[LinkItem] = []
    for idx in top_idx:
        row = links_df.iloc[int(idx)]
        items.append(
            LinkItem(
                id=str(row["id"]),
                name=row["Name"],
                url=row["Link"],
            )
        )
    return items


# ========= GROQ LLM CALL =========
def call_groq_chat(question: str, top_link: str | None) -> str:
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY not configured.")

    system_prompt = (
        "You are ICAway Buddy, a friendly career-learning assistant. "
        "Use the provided ICAway LMS link as the primary resource. "
        "Answer briefly (2–3 sentences max) and then clearly show the link "
        "on its own line so the user can click it."
    )

    user_content = question
    if top_link:
        user_content += f"\n\nMost relevant ICAway course link:\n{top_link}"

    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        "temperature": 0.3,
    }

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }

    resp = requests.post(GROQ_API_URL, headers=headers, json=payload, timeout=60)
    if resp.status_code != 200:
        raise RuntimeError(f"Groq API error {resp.status_code}: {resp.text}")

    data = resp.json()
    content = (
        data.get("choices", [{}])[0]
        .get("message", {})
        .get("content", "")
        .strip()
    )
    return content or "(no answer from LLM)"


# ========= API MODELS =========
class AskRequest(BaseModel):
    question: str
    top_k: int = 3


class SourceItem(BaseModel):
    preview: str
    chunk_id: str
    similarity: float


class AskResponse(BaseModel):
    answer: str
    sources: List[SourceItem]


# ========= ROUTES =========
@app.on_event("startup")
def startup_event():
    load_links_and_embeddings()


@app.get("/health")
def health():
    return {
        "ok": True,
        "num_links": 0 if links_df is None else int(len(links_df)),
        "embed_model": EMBED_MODEL_NAME,
        "groq_configured": GROQ_API_KEY is not None,
    }


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        top_items = cosine_top_k(req.question, k=max(1, req.top_k))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    if not top_items:
        return AskResponse(answer="I couldn't find any ICAway resources for that.", sources=[])

    # pick the single best link
    best = top_items[0]
    top_link = best.url

    # compute similarity scores (normalized dot product already)
    # here we just use a dummy similarity since we already sorted
    sources: List[SourceItem] = [
        SourceItem(preview=item.url, chunk_id=item.id, similarity=1.0 if i == 0 else 0.5)
        for i, item in enumerate(top_items)
    ]

    # call Groq; if it fails, we still show the link
    try:
        answer_text = call_groq_chat(req.question, top_link)
    except Exception as e:
        answer_text = f"Most relevant link:\n{top_link}\n\n(LLM error: {e})"
    else:
        # ensure link is visible even if LLM forgets to show it
        if top_link not in answer_text:
            answer_text += f"\n\nMost relevant link:\n{top_link}"

    return AskResponse(answer=answer_text, sources=sources)


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("server_cloud:app", host="0.0.0.0", port=port)
