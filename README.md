Here are all the modules along with the classes and methods they contain:

* **`ingestion/bm25_store.py`**
* **Class:** `BM25Store`
* **Methods:**
* `__init__(self, index_path: str = "storage/bm25_index.pkl")`
* `build(self, chunks: list[Document])`
* `load(self)`
* `search(self, query: str, top_k: int = 20)`
* `delete_index(self)`




* **`ingestion/embedder.py`**
* **Class:** `DocumentEmbedder`
* **Methods:**
* `__init__(self, model_name: str = "ibm-granite/granite-embedding-small-english-r2")`
* `embed(self, chunks: list[Document])`




* **`ingestion/ingestion_service.py`**
* **Class:** `IngestionService` (This class orchestrates the entire pipeline by calling the other modules)
* **Methods:**
* `__init__(self, vector_store: VectorStore, bm25_store: BM25Store, metadata_store: MetadataStore)`
* `process(self, file: UploadFile)`




* **`ingestion/loder.py`**
* **Class:** `DocumentLoader`
* **Methods:**
* `load(self, file_path: Path)`




* **`ingestion/metadata.py`**
* **Class:** `MetadataStore` (Handles SQLite database interactions for document metadata)
* **Methods:**
* `__init__(self, db_path: str = "storage/metadata.db")`
* `create_table(self)`
* `insert_document(self, document_uuid: str, filename: str, filepath: str, file_hash: str, pages: int, chunks: int, embedding_model: str, status: str = "Indexed")`
* `get_document(self, document_uuid: str)`
* `get_document_by_filename(self, filename: str)`
* `get_document_by_hash(self, file_hash: str)`
* `list_documents(self)`
* `delete_document(self, document_uuid: str)`
* `update_status(self, document_uuid: str, status: str)`
* `close(self)`


* **Class:** `DocumentMetadata` (A python dataclass holding metadata fields)


* **`ingestion/reciever.py`**
* **Class:** `FileReceiver`
* **Methods:**
* `__init__(self)`
* `receive(self, file: UploadFile)`


* **Class:** `UploadedDocument` (A python dataclass representing an uploaded document)


* **`ingestion/splitter.py`**
* **Class:** `DocumentSplitter`
* **Methods:**
* `__init__(self, chunk_size: int = 800, chunk_overlap: int = 150)`
* `split(self, documents: list[Document], doc_uuid: str)`




* **`ingestion/vector_store.py`**
* **Class:** `VectorStore` (Manages connection and actions with Qdrant Client)
* **Methods:**
* `__init__(self, client: QdrantClient, collection_name: str, vector_size: int = 768)`
* `_create_collection(self)`
* `insert(self, chunks: list[Document], embeddings: list[list[float]])`
* `delete(self, document_uuid: str)`
* `search(self, query_vector: list[float], top_k: int = 20)`



Knowledge-Copilot/
├── api/                        # 🌐 FastAPI web endpoints (Routes)
│   ├── __init__.py
│   ├── ingestion_routes.py     # e.g., @app.post("/upload")
│   └── retrieval_routes.py     # e.g., @app.post("/ask")
│
├── services/                   # 🧠 Orchestration & Business Logic
│   ├── __init__.py
│   ├── ingestion_service.py    # (You just moved this here!)
│   └── retrieval_service.py    # (You will build this next)
│
├── ingestion/                  # 📥 Low-level Data Processing modules
│   ├── __init__.py
│   ├── loder.py                # Document loaders
│   ├── splitter.py             # Chunking logic
│   └── embedder.py             # HuggingFace embeddings
│
├── retrieval/                  # 📤 Low-level Query Processing modules
│   ├── __init__.py
│   ├── dense.py                # Qdrant search logic
│   ├── sparse.py               # BM25 search logic
│   └── reranker.py             # Cross-encoder / RRF logic
│
├── stores/                     # 💾 Database / Storage Interfaces
│   ├── __init__.py
│   ├── vector_store.py         # Qdrant interactions
│   ├── bm25_store.py           # Pickle/BM25 interactions
│   └── metadata_store.py       # SQLite interactions
│
├── core/                       # ⚙️ Application Utilities & Config
│   ├── __init__.py
│   ├── config.py               # Environment variables & constants
│   ├── exception.py            # CustomException definitions
│   └── logger.py               # Logging configuration
│
├── schemas/                    # 📄 Pydantic Models for Data Validation
│   ├── __init__.py
│   ├── requests.py             # e.g., AskQuestionRequest
│   └── responses.py            # e.g., IngestionResponse
│
├── storage/                    # 📁 Local Storage (ADD TO .gitignore)
│   ├── uploads/                # Temporarily saved PDFs/Txts
│   ├── qdrant_db/              # Persistent local vector db
│   └── metadata.db             # SQLite database file
│
├── logs/                       # 📝 Log files (ADD TO .gitignore)
├── tests/                      # 🧪 Automated Tests
│   ├── __init__.py
│   ├── test_ingestion.py       # (Move your test script here!)
│   └── test_retrieval.py
│
├── requirements.txt
├── main.py                     # 🚀 The FastAPI Application Entrypoint
└── README.md



📱 LAYER 1: The Client (Streamlit UI)
      │ 
      │ (Sends HTTP Request over the internet: "What is the Copilot?")
      ▼
🌐 LAYER 2: The API Routes (FastAPI)
      │   - Acts as the receptionist. 
      │   - Checks if the JSON payload is valid.
      │   - Immediately hands the question to the Service layer.
      ▼
🧠 LAYER 3: The Services (IngestionService / RetrievalService)
      │   - Acts as the Manager.
      │   - Says: "Okay, I need to get chunks, rerank them, and ask the LLM."
      │   - It delegates the actual heavy lifting to the workers below.
      ▼
⚙️ LAYER 4: The Workers & Models (Dense, Sparse, Cross-Encoder, LLM)
      │   - The IBM Granite model turns the question into math.
      │   - The Ettin model reads and scores the text.
      │   - The LLM writes the final English answer.
      ▼
💾 LAYER 5: The Data Stores (Qdrant, BM25, SQLite)
          - The filing cabinets where the actual text and vectors live.