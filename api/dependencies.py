from qdrant_client import QdrantClient
from ingestion.vector_store import VectorStore
from ingestion.bm25_store import BM25Store
from ingestion.metadata import MetadataStore
from ingestion.embedder import DocumentEmbedder
from services.ingestion_service import IngestionService
from retrieval.dense import DenseRetriever
from retrieval.sparse import SparseRetriever
from retrieval.reranker import RRFReranker
from retrieval.cross_encoder import EntropyReranker
from services.retrieval_service import RetrievalService
from typing import Optional, List, Any
from langchain_openai import ChatOpenAI

# 1. Initialize Stores (Saved to disk so data persists between server restarts!)
print("Loading Databases...")
qdrant_client = QdrantClient(path="storage/qdrant_db")
vector_store = VectorStore(client=qdrant_client, collection_name="knowledge_base", vector_size=384)
bm25_store = BM25Store(index_path="storage/bm25_index.pkl")
metadata_store = MetadataStore(db_path="storage/metadata.db")

# 2. Initialize Models & Services
print("Loading ML Models into RAM...")
embedder = DocumentEmbedder()

# Ingestion
ingestion_service = IngestionService(vector_store, bm25_store, metadata_store)

# Retrieval
dense_retriever = DenseRetriever(embedder, vector_store)
sparse_retriever = SparseRetriever(bm25_store)
rrf = RRFReranker(k=60)
entropy = EntropyReranker(model_name="cross-encoder/ettin-reranker-32m-v1")


liquid_llm = ChatOpenAI(
    base_url="http://localhost:1234/v1", 
    api_key="lm-studio", # It doesn't matter what you put here, but it can't be blank!
    temperature=0.3,
    max_tokens=512
)

retrieval_service = RetrievalService(
    dense_retriever=dense_retriever,
    sparse_retriever=sparse_retriever,
    rrf_reranker=rrf,
    entropy_reranker=entropy,
    llm=liquid_llm
)

# Dependency Injection Functions
def get_ingestion_service():
    return ingestion_service

def get_retrieval_service():
    return retrieval_service

def get_metadata_store():
    return metadata_store