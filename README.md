# Knowledge Copilot

A fully local, high-performance Retrieval-Augmented Generation (RAG) backend and API. This project implements a modern hybrid search pipeline with dual-stage reranking, designed to run sophisticated open-source LLMs entirely locally using pure Python and PyTorch—completely bypassing C++ compiler errors and hardware incompatibilities.

## 🚀 Features

* **Hybrid Search Retrieval:** Combines the semantic understanding of Dense Vector Search (Qdrant) with the exact-keyword matching of Sparse Lexical Search (BM25) for ultimate retrieval recall.
* **Dual-Stage Reranking:** Implements Reciprocal Rank Fusion (RRF) to seamlessly merge dense and sparse results, followed by a final precision pass using the lightning-fast `cross-encoder/ettin-reranker-32m-v1`.
* **100% Local Inference:** Integrates `HuggingFacePipeline` and PyTorch to run powerful reasoning models (like `google/gemma-2-2b-it` or `LiquidAI/LFM2.5-230M`) directly in system RAM.
* **Persistent Storage:** Utilizes Qdrant (saved to disk) for vector embeddings, pickle files for the BM25 index, and SQLite for robust document metadata management.
* **Modern Package Management:** Built and managed with `uv` for blazingly fast dependency resolution.

---

## 📂 Project Structure

```text
📁 Knowledge-Copilot/
 ├── 📁 api/
 │    ├── 📄 dependencies.py       # Core service initialization and LLM orchestration
 │    ├── 📄 ingestion_routes.py   # FastAPI endpoints for uploading documents
 │    └── 📄 retrieval_routes.py   # FastAPI endpoints for RAG queries
 ├── 📁 core/                      # Application configuration and core settings
 ├── 📁 ingestion/
 │    ├── 📄 bm25_store.py         # Sparse retrieval logic
 │    ├── 📄 embedder.py           # Document embedding generation
 │    ├── 📄 metadata.py           # SQLite metadata tracking
 │    └── 📄 vector_store.py       # Qdrant client interactions
 ├── 📁 retrieval/
 │    ├── 📄 cross_encoder.py      # EntropyReranker (Ettin 32M)
 │    ├── 📄 dense.py              # Semantic search retrieval
 │    ├── 📄 reranker.py           # Reciprocal Rank Fusion (RRF) logic
 │    └── 📄 sparse.py             # Keyword search retrieval
 ├── 📁 services/
 │    ├── 📄 ingestion_service.py  # Orchestrates document chunking and saving
 │    └── 📄 retrieval_service.py  # Orchestrates the hybrid search and LLM synthesis
 ├── 📁 storage/                   # Persistent local database files (Qdrant, SQLite, BM25)
 ├── 📄 app.py                     # Streamlit frontend user interface
 ├── 📄 main.py                    # FastAPI application entry point
 ├── 📄 requirements.txt           # Standard dependency list
 └── 📄 .env                       # Environment variables (e.g., HF_TOKEN)

```

---

## 🛠️ Installation & Setup

This project uses the `uv` package manager for ultra-fast, reliable installation without the headaches of legacy Python environments.

**1. Clone the repository and navigate to the root directory**

```bash
git clone https://github.com/yourusername/Knowledge-Copilot.git
cd Knowledge-Copilot

```

**2. Install dependencies using `uv**`

```bash
# Install the core local PyTorch and Hugging Face libraries
uv add torch transformers langchain-huggingface qdrant-client fastapi uvicorn streamlit python-dotenv

```

**3. Configure your Environment variables**
Create a `.env` file in the root directory and add your Hugging Face token (required to download gated PyTorch models like Gemma 2 directly to your local cache).

```env
HF_TOKEN=hf_your_secure_token_here

```

---

## 🧠 Model Configuration

The system is pre-configured in `api/dependencies.py` to use a pure-Python PyTorch pipeline. The default configuration uses the highly capable `google/gemma-2-2b-it`.

If you are running on a machine with limited RAM (under 8GB), you can swap the `repo_id` to a micro-model like `LiquidAI/LFM2.5-230M` for lower memory footprint (though reasoning capabilities will be reduced).

```python
# api/dependencies.py
from transformers import pipeline

pipe = pipeline(
    "text-generation", 
    model="google/gemma-2-2b-it", # Change repo_id here if needed
    max_new_tokens=512,
    temperature=0.1, 
    top_k=50,
    repetition_penalty=1.05,
    do_sample=True,
    device="cpu" 
)

```

---

## ⚡ Running the Application

The architecture is separated into a FastAPI backend and a Streamlit frontend. You will need two separate terminal windows.

**Terminal 1: Start the FastAPI Backend**

```bash
uvicorn main:app --reload

```

*(Note: On the first run, the backend will take a few moments to download the LLM and Reranker weights into your local Hugging Face cache.)*

**Terminal 2: Start the Streamlit Frontend**

```bash
streamlit run app.py

```

Navigate to the `localhost` URL provided by Streamlit to begin uploading documents and interacting with your fully local RAG Copilot!

<img width="676" height="831" alt="1cbc2f1834104369bef39f4b58fc9e1e" src="https://github.com/user-attachments/assets/c7211b4d-0f1f-4e74-ad5f-598c3417c649" />
