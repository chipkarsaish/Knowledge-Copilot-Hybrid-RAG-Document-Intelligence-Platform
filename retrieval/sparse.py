from core.logger import logging
from core.exception import CustomException
from ingestion.bm25_store import BM25Store

class SparseRetriever:
    def __init__(self, bm25_store: BM25Store):
        self.bm25_store = bm25_store

    def retrieve(self, query: str, top_k: int = 10) -> list[dict]:
        logging.info(f"Performing Sparse (BM25) Search for query: '{query}'")
        try:
            results = self.bm25_store.search(query=query, top_k=top_k)
            
            # Format results
            formatted_results = []
            for doc in results:
                formatted_results.append({
                    "text": doc.page_content,
                    "metadata": doc.metadata,
                    "sparse_score": doc.metadata.get("bm25_score", 0.0)
                })
                
            return formatted_results
        except Exception as e:
            raise CustomException(e)