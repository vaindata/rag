from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.http import models
from fastembed import SparseTextEmbedding
from uuid import uuid4
import os, time, pickle, warnings
import tqdm
from dotenv import load_dotenv

warnings.filterwarnings("ignore")
load_dotenv()

qdrant_url    = os.getenv("QDRANT_URL")
qdrant_api_key = os.getenv("QDRANT_API_KEY")

# model_name = "Qwen/Qwen3-Embedding-0.6B"
model_name = "infly/inf-retriever-v1-1.5b"

# ── CPU-safe settings: small encode batch to avoid OOM ──
embeddings = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs={"device": "cpu"},
    encode_kwargs={
        "batch_size": 8,          # keep RAM usage low on CPU
        "normalize_embeddings": True,
    },
)
print(f"[Info] Model loaded: {model_name}")

dimension = len(embeddings.embed_query("test"))
print(f"[Info] Embedding dimension: {dimension}")

client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key, timeout=800)

COLLECTION_NAME  = f"insurance_hybrid_{model_name.split('/')[-1].replace('-', '_')}"
UPSERT_BATCH     = 64    # safe for Qdrant Cloud; raise to 128 if no timeout errors
EMBED_BATCH      = 8     # texts per dense-embed call — controls peak RAM
BM25_BATCH       = 128   # fastembed is C++, handles larger batches fine on CPU


def ensure_collection_hybrid(client, collection_name, dense_size):
    if collection_name in [c.name for c in client.get_collections().collections]:
        print(f"ℹ️  Collection '{collection_name}' already exists.")
        return
    client.create_collection(
        collection_name=collection_name,
        vectors_config={
            "dense": models.VectorParams(size=dense_size, distance=models.Distance.COSINE),
        },
        sparse_vectors_config={
            "bm25": models.SparseVectorParams(modifier=models.Modifier.IDF),
        },
    )

ensure_collection_hybrid(client, COLLECTION_NAME, dense_size=dimension)

with open("docs.pkl", "rb") as f:
    docs = pickle.load(f)

texts = [doc.page_content for doc in docs]
print(f"[Info] Loaded {len(texts)} documents.")

# ── BM25 first — pure C++, fast, low memory ──
print("[Info] Computing BM25 sparse vectors…")
bm25_model      = SparseTextEmbedding("Qdrant/bm25")
bm25_sparse_list = list(
    tqdm.tqdm(
        bm25_model.passage_embed(texts, batch_size=BM25_BATCH),
        total=len(texts), desc="BM25"
    )
)

# ── Stream dense embeddings + upsert in chunks to cap peak RAM ──
# Instead of embedding ALL texts then building ALL points (doubles RAM),
# we embed a chunk → build points → upsert → discard before the next chunk.

print("[Info] Embedding dense vectors and upserting in streaming chunks…")
start_time = time.time()

for chunk_start in tqdm.tqdm(range(0, len(docs), UPSERT_BATCH), desc="Chunks"):
    chunk_end   = min(chunk_start + UPSERT_BATCH, len(docs))
    chunk_texts = texts[chunk_start:chunk_end]
    chunk_docs  = docs[chunk_start:chunk_end]

    # embed only this chunk (EMBED_BATCH controls inner batching)
    chunk_dense = embeddings.embed_documents(chunk_texts)

    points = []
    for i, doc in enumerate(chunk_docs):
        bm25_obj = bm25_sparse_list[chunk_start + i].as_object()
        points.append(
            models.PointStruct(
                id=str(uuid4()),
                vector={
                    "dense": chunk_dense[i],
                    "bm25": models.SparseVector(
                        indices=bm25_obj["indices"],
                        values=bm25_obj["values"],
                    ),
                },
                payload={"text": doc.page_content, **doc.metadata},
            )
        )

    client.upsert(collection_name=COLLECTION_NAME, points=points)
    # chunk_dense and points go out of scope here → GC reclaims RAM

end_time = time.time()
print(f"[Info] Done! Total time: {end_time - start_time:.2f}s")