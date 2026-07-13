import time
import requests
from sentence_transformers import SentenceTransformer

API_URL = "http://127.0.0.1:8000/query"
TEST_QUESTION = "How does semantic chunking work?"


def old_way_reload_every_time(question, n_runs=3):
    """Simulates query.py's behavior: model is loaded fresh each call."""
    print("── OLD WAY: reload model on every request ──")
    times = []
    for i in range(n_runs):
        start = time.time()
        model = SentenceTransformer("all-MiniLM-L6-v2")  # reloaded every time
        _ = model.encode(question)
        elapsed = time.time() - start
        times.append(elapsed)
        print(f"  Run {i+1}: {elapsed:.2f}s")
    avg = sum(times) / len(times)
    print(f"  → average: {avg:.2f}s\n")
    return avg


def new_way_via_api(question, n_runs=3):
    """Model loaded once at server startup; measures cache miss vs hit."""
    print("── NEW WAY: FastAPI server (model loaded once + caching) ──")
    times = []
    for i in range(n_runs):
        start = time.time()
        resp = requests.post(API_URL, json={"question": question, "top_k": 3})
        elapsed = time.time() - start
        data = resp.json()
        cached = data.get("cached")
        times.append(elapsed)
        print(f"  Run {i+1}: {elapsed*1000:.1f}ms (cached={cached}, "
              f"server-reported latency={data.get('latency_ms')}ms)")
    avg = sum(times) / len(times)
    print(f"  → average: {avg*1000:.1f}ms\n")
    return avg


if __name__ == "__main__":
    print(f"Test question: '{TEST_QUESTION}'\n")

    old_avg = old_way_reload_every_time(TEST_QUESTION)
    try:
        new_avg = new_way_via_api(TEST_QUESTION)
        speedup = old_avg / new_avg if new_avg > 0 else float("inf")
        print(f"📊 RESULT: API is ~{speedup:.1f}x faster than the reload-every-time approach")
        print("   (first API call = cache miss, later calls = cache hit and near-instant)")
    except requests.exceptions.ConnectionError:
        print("⚠️  Couldn't reach the API — start it first with:")
        print("   uvicorn week4_api_speedup:app --reload")