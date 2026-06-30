import asyncio
import os
from pathlib import Path
from fastapi import UploadFile
from qdrant_client import QdrantClient

# Store Imports
from ingestion.vector_store import VectorStore
from ingestion.bm25_store import BM25Store
from ingestion.metadata import MetadataStore

# Ingestion Imports
from services.ingestion_service import IngestionService

# Retrieval Imports
from retrieval.dense import DenseRetriever
from retrieval.sparse import SparseRetriever
from retrieval.reranker import RRFReranker
from retrieval.cross_encoder import EntropyReranker
from services.retrieval_service import RetrievalService

# LangChain Mock LLM (For testing without an API key)
from langchain_core.language_models.llms import LLM
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from typing import Optional, List, Any

class MockLLM(LLM):
    def _call(self, prompt: str, stop: Optional[List[str]] = None, run_manager: Optional[CallbackManagerForLLMRun] = None, **kwargs: Any) -> str:
        # We pretend the LLM read the context and output this exact JSON
        return """
        {
            "answer": "The Knowledge Copilot is a Hybrid RAG Document Intelligence Platform that uses Qdrant and BM25.",
            "confidence_score": 0.98,
            "used_source_ids": ["source_1", "source_2"]
        }
        """
    @property
    def _llm_type(self) -> str:
        return "mock_llm"

async def run_test():
    print("🚀 Starting Hybrid RAG Retrieval Test...\n")

    # 1. SETUP DATABASES
    print("Setting up in-memory databases...")
    qdrant_client = QdrantClient(":memory:") 
    vector_store = VectorStore(client=qdrant_client, collection_name="test_collection", vector_size=384)
    bm25_store = BM25Store(index_path="storage/test_bm25_index.pkl")
    metadata_store = MetadataStore(db_path="storage/test_metadata.db")

    # 2. QUICK INGESTION (So we have data to search)
    test_filepath = "sample_test.txt"
    if not os.path.exists(test_filepath):
        with open(test_filepath, "w") as f:
            f.write("The Knowledge Copilot is a Hybrid RAG Document Intelligence Platform. It uses Qdrant for dense vector search and BM25 for sparse keyword search.")
    
    print("Ingesting sample document...")
    ingestion_service = IngestionService(vector_store, bm25_store, metadata_store)
    with open(test_filepath, "rb") as f:
        upload_file = UploadFile(filename=test_filepath, file=f)
        await ingestion_service.process(upload_file)

    # 3. INITIALIZE RETRIEVAL PIPELINE
    print("\nInitializing Retrieval Pipeline (Downloading Ettin model if first run)...")
    
    # You need the same embedder for Dense retrieval
    from ingestion.embedder import DocumentEmbedder
    embedder = DocumentEmbedder()

    dense_retriever = DenseRetriever(embedder=embedder, vector_store=vector_store)
    sparse_retriever = SparseRetriever(bm25_store=bm25_store)
    rrf_reranker = RRFReranker(k=60)
    entropy_reranker = EntropyReranker(model_name="cross-encoder/ettin-reranker-32m-v1")
    
    # Initialize the Service with our Mock LLM
    mock_llm = MockLLM()
    retrieval_service = RetrievalService(
        dense_retriever=dense_retriever,
        sparse_retriever=sparse_retriever,
        rrf_reranker=rrf_reranker,
        entropy_reranker=entropy_reranker,
        llm=mock_llm
    )

    # 4. TEST THE PIPELINE
    query = "What is the Knowledge Copilot?"
    print(f"\n❓ Asking Question: '{query}'")
    print("-" * 50)
    
    result = retrieval_service.answer_query(query)

    # 5. PRINT RESULTS
    print("\n✅ Final Pipeline Output:")
    print(f"Answer: {result['answer']}")
    print(f"Confidence: {result['confidence_score']}")
    print("Verified Citations:")
    for citation in result['citations']:
        print(f"  - File: {citation['filename']}, Cross-Encoder Score: {citation['cross_encoder_score']:.4f}")

    print("-" * 50)

if __name__ == "__main__":
    Path("storage/uploads").mkdir(parents=True, exist_ok=True)
    asyncio.run(run_test())