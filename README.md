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



