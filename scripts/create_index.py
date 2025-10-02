# scripts/create_index.py
import os, json
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from tqdm import tqdm

os.makedirs("data", exist_ok=True)

LIT_FILE = "data/literature.jsonl"
METADATA_FILE = "data/metadata.json"
INDEX_FILE = "data/faiss_index.bin"

print("Loading literature...")
docs = []
with open(LIT_FILE, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            docs.append(json.loads(line))

texts = [d["text"] for d in docs]
print(f"Loaded {len(texts)} docs.")

print("Loading embedding model (sentence-transformers)...")
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

print("Computing embeddings...")
embeddings = embed_model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
# convert to float32
embeddings = embeddings.astype("float32")

# normalize for cosine similarity via dot product
faiss.normalize_L2(embeddings)

d = embeddings.shape[1]
index = faiss.IndexFlatIP(d)  # inner-product on normalized vectors = cosine similarity
index.add(embeddings)
print(f"Index has {index.ntotal} vectors, dimension {d}")

print("Saving index and metadata...")
faiss.write_index(index, INDEX_FILE)
with open(METADATA_FILE, "w", encoding="utf-8") as f:
    json.dump(docs, f, ensure_ascii=False, indent=2)

print("Done. Files created:", METADATA_FILE, INDEX_FILE)
