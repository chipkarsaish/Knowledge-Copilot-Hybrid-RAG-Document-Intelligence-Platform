from core.logger import logging
from core.exception import CustomException
from ingestion.embedder import DocumentEmbedder
from ingestion.vector_store import VectorStore

class DenseRetriever:
    def __init__(self, embedder: DocumentEmbedder, vector_store: VectorStore):
        self.embedder = embedder
        self.vector_store = vector_store

    def retrieve(self, query: str, top_k: int = 10) -> list[dict]:
        logging.info(f"Performing Dense Search for query: '{query}'")
        try:
            # The embedder expects a LangChain Document list, but we can access the underlying HuggingFace model directly for a single string
            query_embedding = self.embedder.embedding_model.embed_query(query)
            
            # Search Qdrant
            results = self.vector_store.search(query_vector=query_embedding, top_k=top_k)
            
            # Format results into a standard dictionary
            formatted_results = []
            for point in results:
                formatted_results.append({
                    "id": point.id,
                    "text": point.payload.get("text", ""),
                    "metadata": point.payload,
                    "dense_score": point.score
                })
            
            return formatted_results
        except Exception as e:
            raise CustomException(e)