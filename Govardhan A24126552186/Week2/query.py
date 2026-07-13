import chromadb
from sentence_transformers import SentenceTransformer

# ─── SETUP ─────────────────────────────────────────────────────────
def load_collection():
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_collection("algoverse_docs")
    print(f"✅ Connected to ChromaDB")
    return collection

# ─── QUERY ─────────────────────────────────────────────────────────
def ask(question, collection, top_k=3):
    print(f"\n❓ Question: {question}")
    print("─" * 50)

    # Convert question to vector
    model = SentenceTransformer('all-MiniLM-L6-v2')
    question_vector = model.encode(question).tolist()

    # Search ChromaDB for most similar chunks
    results = collection.query(
        query_embeddings=[question_vector],
        n_results=top_k
    )

    # Show results
    docs      = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    for i, (doc, meta, dist) in enumerate(zip(docs, metadatas, distances)):
        similarity = round((1 - dist) * 100, 1)  # convert distance to % similarity
        print(f"\n📄 Result {i+1} — Page {meta['page']} ({similarity}% match)")
        print(f"   {doc}")

    return docs

# ─── RUN ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    collection = load_collection()

    print("\n💬 Ask anything about your PDF.")
    print("   Type 'quit' to exit.\n")

    while True:
        question = input("Your question: ").strip()

        if question.lower() in ["quit", "exit", "q"]:
            print("👋 Bye!")
            break

        if question == "":
            print("⚠️  Please type a question first.\n")
            continue

        ask(question, collection)