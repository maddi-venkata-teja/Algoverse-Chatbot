import time
import chromadb
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from sentence_transformers import SentenceTransformer
from cachetools import TTLCache

# Optional: use the reranker you trained in Week 3, if it exists
try:
    from gradient_accumulation import rerank
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    RERANKER_AVAILABLE = True
except Exception:
    RERANKER_AVAILABLE = False


app = FastAPI(title="Algoverse RAG API", version="1.0")

# ─── SPEED-UP #1: Load everything ONCE at startup ───────────────────
print("🚀 Starting up — loading model + DB connection once...")
_start = time.time()

embed_model = SentenceTransformer("all-MiniLM-L6-v2")

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_collection("algoverse_docs")

reranker_model, reranker_tokenizer = None, None
if RERANKER_AVAILABLE:
    try:
        reranker_tokenizer = AutoTokenizer.from_pretrained("./reranker_model")
        reranker_model = AutoModelForSequenceClassification.from_pretrained(
            "./reranker_model", num_labels=1
        )
        reranker_model.eval()
        print("  → Week 3 reranker loaded, will be used to re-score results")
    except Exception:
        print("  → No trained reranker found yet, skipping (run Week 3 script first)")

print(f"✅ Startup complete in {time.time() - _start:.2f}s\n")

# ─── SPEED-UP #2: Cache recent query results for 10 minutes ─────────
# maxsize=200 -> keeps the 200 most recent distinct questions cached
query_cache = TTLCache(maxsize=200, ttl=600)


class QueryRequest(BaseModel):
    question: str
    top_k: int = 3
    use_reranker: bool = False


class BatchQueryRequest(BaseModel):
    questions: List[str]
    top_k: int = 3


def _search(question: str, top_k: int, use_reranker: bool = False):
    question_vector = embed_model.encode(question).tolist()

    # Over-fetch a few extra candidates if we're going to rerank
    n_results = top_k * 3 if (use_reranker and reranker_model) else top_k

    results = collection.query(query_embeddings=[question_vector], n_results=n_results)
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    dists = results["distances"][0]

    if use_reranker and reranker_model:
        reranked = rerank(question, docs, model=reranker_model,
                           tokenizer=reranker_tokenizer, top_k=top_k)
        return [{"text": text, "score": round(score, 3)} for score, text in reranked]

    matches = []
    for doc, meta, dist in zip(docs[:top_k], metas[:top_k], dists[:top_k]):
        similarity = round((1 - dist) * 100, 1)
        matches.append({"text": doc, "page": meta["page"], "similarity": similarity})
    return matches


# ─── SPEED-UP #4: async endpoint ─────────────────────────────────────
@app.post("/query")
async def query(req: QueryRequest):
    cache_key = (req.question.strip().lower(), req.top_k, req.use_reranker)

    if cache_key in query_cache:
        t0 = time.time()
        result = query_cache[cache_key]
        return {"cached": True, "latency_ms": round((time.time() - t0) * 1000, 2), "results": result}

    t0 = time.time()
    result = _search(req.question, req.top_k, req.use_reranker)
    latency_ms = round((time.time() - t0) * 1000, 2)

    query_cache[cache_key] = result
    return {"cached": False, "latency_ms": latency_ms, "results": result}


# ─── SPEED-UP #3: batch encoding for multiple questions at once ─────
@app.post("/query_batch")
async def query_batch(req: BatchQueryRequest):
    t0 = time.time()

    # Encode ALL questions in a single batched call instead of a loop
    vectors = embed_model.encode(req.questions).tolist()

    all_results = []
    for question, vector in zip(req.questions, vectors):
        results = collection.query(query_embeddings=[vector], n_results=req.top_k)
        docs = results["documents"][0]
        metas = results["metadatas"][0]
        matches = [{"text": d, "page": m["page"]} for d, m in zip(docs, metas)]
        all_results.append({"question": question, "results": matches})

    return {"latency_ms": round((time.time() - t0) * 1000, 2), "answers": all_results}


@app.get("/health")
async def health():
    return {"status": "ok", "cache_size": len(query_cache), "reranker_loaded": reranker_model is not None}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)