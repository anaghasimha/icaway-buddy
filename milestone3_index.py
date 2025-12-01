import os
from typing import List, Dict
import shutil
import numpy as np
import math

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from ingestion import build_page_chunks, PAGE_ID


# ---------------------------
# 0. config
# ---------------------------

PERSIST_DIR = "./chroma_db"
COLLECTION_NAME = "notion_chunks"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


# ---------------------------
# 1. Embedding model wrapper
# ---------------------------

class LocalEmbeddingModel:
    """
    We are no longer letting Chroma call the embedder.
    We call the embedder ourselves.

    This class is ONLY for us (not passed into Chroma).
    """

    def __init__(self, model_name: str = EMBED_MODEL):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_sentence_embedding_dimension()

    def encode_batch(self, texts: List[str]) -> np.ndarray:
        # Clean inputs
        clean = []
        for t in texts:
            if t is None:
                clean.append("")
            elif isinstance(t, str):
                clean.append(t.strip())
            else:
                clean.append(str(t).strip())

        embs = self.model.encode(
            clean,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )  # shape: (len(texts), dim)
        return embs

    def encode_one(self, text: str) -> np.ndarray:
        return self.encode_batch([text])[0]


def get_chroma_collection(reset_if_conflict: bool = True):
    """
    We STILL store in Chroma, but we will NOT rely on Chroma to run the embedder.
    We'll manage embeddings ourselves.
    """

    try:
        chroma_client = chromadb.PersistentClient(
            path=PERSIST_DIR,
            settings=Settings(anonymized_telemetry=False)
        )

        # We create/get the collection WITHOUT giving Chroma an embedding_function.
        # That means we take responsibility for embeddings.
        collection = chroma_client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"source": "notion"}
        )
        return collection

    except Exception as e:
        if reset_if_conflict:
            print("Chroma threw an error. Resetting local DB...")
            print("Details:", repr(e))
            shutil.rmtree(PERSIST_DIR, ignore_errors=True)

            chroma_client = chromadb.PersistentClient(
                path=PERSIST_DIR,
                settings=Settings(anonymized_telemetry=False)
            )
            collection = chroma_client.get_or_create_collection(
                name=COLLECTION_NAME,
                metadata={"source": "notion"}
            )
            return collection
        else:
            raise


# ---------------------------
# 2. Indexing logic
# ---------------------------

def index_page(page_id: str, collection, embedder: LocalEmbeddingModel) -> None:
    """
    1. Build chunks from Notion (Milestone 2)
    2. Locally embed each chunk text with sentence-transformers
    3. Upsert into Chroma with *explicit* embeddings
    """
    chunks: List[Dict[str, str]] = build_page_chunks(page_id)

    if not chunks:
        print("No chunks found for this page.")
        return

    ids = []
    docs = []
    metas = []
    vectors = []

    # generate embeddings once here
    texts = [c["text"] for c in chunks]
    embs = embedder.encode_batch(texts)  # shape (n_chunks, dim)

    for i, c in enumerate(chunks):
        ids.append(c["chunk_id"])
        docs.append(c["text"])
        metas.append({
            "page_id": c["page_id"],
            "source_block_ids": ",".join(c["source_block_ids"]),
        })
        vectors.append(embs[i].tolist())

    collection.upsert(
        ids=ids,
        documents=docs,
        metadatas=metas,
        embeddings=vectors
    )

    print(f"Indexed {len(chunks)} chunks from page {page_id} into local DB at {PERSIST_DIR}.")


# ---------------------------
# 3. Manual semantic search
# ---------------------------

def get_all_chunks(collection):
    """
    Pull EVERYTHING out of the collection:
    ids, docs, metas, embeddings.

    We have to page through because Chroma may limit batch size.
    We'll just pull in batches of 100.
    """
    all_ids = []
    all_docs = []
    all_metas = []
    all_embs = []

    offset = 0
    page_size = 100

    while True:
        res = collection.get(
            ids=None,
            where=None,
            limit=page_size,
            offset=offset,
            include=["embeddings", "metadatas", "documents"]
        )

        batch_ids = res.get("ids", [])
        if not batch_ids:
            break  # no more data

        all_ids.extend(batch_ids)
        all_docs.extend(res.get("documents", []))
        all_metas.extend(res.get("metadatas", []))
        all_embs.extend(res.get("embeddings", []))

        offset += len(batch_ids)

    # convert embeddings to numpy array
    all_embs = np.array(all_embs, dtype=float)
    return all_ids, all_docs, all_metas, all_embs


def cosine_sim_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarity between every row in a and every row in b.
    a: (N, d)
    b: (M, d)
    return (N, M)
    """
    # normalize just to be extra safe
    def _norm(x):
        denom = np.linalg.norm(x, axis=1, keepdims=True)
        denom[denom == 0.0] = 1.0
        return x / denom

    a_n = _norm(a)
    b_n = _norm(b)
    return a_n @ b_n.T  # (N,d) @ (d,M) -> (N,M)


def manual_search(query: str, collection, embedder: LocalEmbeddingModel, top_k: int = 4):
    """
    Our own retrieval:
    - get all stored chunk embeddings from Chroma
    - embed the query locally
    - compute cosine sim
    - return top_k matches
    """
    ids, docs, metas, embs = get_all_chunks(collection)

    if len(ids) == 0:
        return []

    q_vec = embedder.encode_one(query)  # shape (dim,)
    q_vec = q_vec.reshape(1, -1)        # shape (1, dim)

    sims = cosine_sim_matrix(embs, q_vec).reshape(-1)  # shape (num_chunks,)
    # higher cosine sim = more similar
    top_idx = np.argsort(-sims)[:top_k]

    results = []
    for idx in top_idx:
        results.append({
            "chunk_id": ids[idx],
            "text": docs[idx],
            "metadata": metas[idx],
            "similarity": float(sims[idx]),
        })
    return results


# ---------------------------
# 4. Manual smoke test
# ---------------------------

if __name__ == "__main__":
    # Step 0: init embedder (local, no API cost)
    embedder = LocalEmbeddingModel()

    # Step 1: open (or recreate) Chroma collection
    collection = get_chroma_collection()

    # Step 2: index the Notion page into Chroma (this will overwrite/upsert same IDs)
    index_page(PAGE_ID, collection, embedder)

    # Step 3: ask a question using our own retrieval logic
    user_question = "What am I supposed to do in this Notion workspace?"
    hits = manual_search(user_question, collection, embedder, top_k=3)

    print("\nTop matches:\n")
    for h in hits:
        print("----")
        print("chunk_id:", h["chunk_id"])
        print("from page:", h["metadata"]["page_id"])
        print("source blocks:", h["metadata"]["source_block_ids"])
        print("similarity score (higher = closer):", h["similarity"])
        preview = h["text"][:400]
        print(preview + ("..." if len(h["text"]) > 400 else ""))
        print()
