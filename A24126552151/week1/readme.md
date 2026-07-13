A local, offline Retrieval-Augmented Generation (RAG) assistant tailored for Data Structures & Algorithms (DSA), featuring Streamlit, LangChain, Ollama, ChromaDB, and Docker deployment.

## 🏗️ Week 1: System Architecture

### 1. Component Workflow
* **Frontend:** Streamlit Chat Interface, Algorithm Visualizer, Dashboard.
* **Orchestration:** LangChain managing `ConversationBufferMemory` and `RetrievalQA`.
* **Knowledge Base:** ChromaDB vector database storing chunked, metadata-tagged DSA tutorials.
* **Inference Engine:** Ollama running quantized `Llama 3 8B` or `Mistral 7B` locally.

### 2. Infrastructure Specifications
* **Development Machine:** 6-Core CPU, 16 GB RAM, 100 GB SSD, Ubuntu 22.04.
* **Fine-Tuning Environment:** NVIDIA RTX 3060 (12GB VRAM minimum) / A100, QLoRA pipeline.
* **Production Cluster:** 8-Core CPU, 32 GB RAM, RTX 4060 GPU, NGINX Reverse Proxy.

### 3. Target Metrics & Performance
* **LLM Generation:** < 3.0 seconds response latency.
* **Vector Retrieval:** < 500 ms database lookup time.
* **Guardrails:** > 90% accuracy target with < 5% hallucination limit.
* **Observability:** Prometheus, Grafana, and Loki container monitoring.

## 🛠️ Verification Setup

Verify your host machine infrastructure meets the deployment requirements outlined above.

```bash
# Clone the repository
git clone <your-repository-url>
cd dsa-ai-assistant

# Run the environment checking script
python scripts/system_check.py
```