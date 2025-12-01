import os
import pandas as pd
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

# ============= CONFIG =============
EXCEL_PATH = "/Users/anaghamv/Downloads/icaway_lms_links.xlsx"  # your Excel file
CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "notion_links"  # same as before so your server works
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
# =================================

def main():
    # Load Excel
    if not os.path.exists(EXCEL_PATH):
        print(f"❌ File '{EXCEL_PATH}' not found.")
        return

    df = pd.read_excel(EXCEL_PATH)
    if "Name" not in df.columns or "Link" not in df.columns:
        print("❌ Excel must have columns: 'Name' and 'Link'")
        return

    # Initialize embedding + chroma
    embedder = SentenceTransformer(EMBED_MODEL_NAME)
    chroma_client = chromadb.PersistentClient(
        path=CHROMA_DIR,
        settings=Settings(anonymized_telemetry=False)
    )

    try:
        collection = chroma_client.get_collection(name=COLLECTION_NAME)
    except Exception:
        collection = chroma_client.create_collection(name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"})

    ids, docs, metas, embs = [], [], [], []

    for idx, row in df.iterrows():
        name = str(row["Name"]).strip()
        url = str(row["Link"]).strip()
        if not url or not url.startswith("http"):
            continue

        text = f"{name}\n{url}"
        emb = embedder.encode([text])[0].tolist()

        ids.append(str(idx))
        docs.append(text)
        metas.append({"title": name, "url": url})
        embs.append(emb)

    if not ids:
        print("❌ No valid rows found.")
        return

    collection.add(
        ids=ids,
        documents=docs,
        metadatas=metas,
        embeddings=embs,
    )

    print(f"✅ Ingested {len(ids)} links into '{COLLECTION_NAME}'.")
    print("Done.")

if __name__ == "__main__":
    main()
