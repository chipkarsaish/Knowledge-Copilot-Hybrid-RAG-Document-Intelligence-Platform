from core.logger import logging
from sentence_transformers import CrossEncoder
from core.exception import CustomException

class EntropyReranker:
    def __init__(self, model_name: str = "cross-encoder/ettin-reranker-32m-v1"):
        logging.info(f"Loading Cross-Encoder model: {model_name}")
        # Initialize the lightweight Ettin model
        # device="cpu" is great for this model since it is so small!
        self.model = CrossEncoder(model_name, device="cpu", max_length=512)

    def rerank(self, query: str, documents: list[dict], top_n: int = 5) -> list[dict]:
        logging.info(f"Applying Cross-Encoder Reranking to {len(documents)} documents")
        try:
            if not documents:
                return []

            # Format inputs for the CrossEncoder: list of pairs [(query, doc_text1), (query, doc_text2), ...]
            pairs = [[query, doc["text"]] for doc in documents]
            
            # Predict scores using the neural network
            scores = self.model.predict(pairs)

            # Attach scores to the documents
            for i, doc in enumerate(documents):
                doc["cross_encoder_score"] = float(scores[i])

            # Sort by the new score (highest first)
            ranked_docs = sorted(documents, key=lambda x: x["cross_encoder_score"], reverse=True)

            # Return the exact number requested for the prompt builder
            return ranked_docs[:top_n]
            
        except Exception as e:
            raise CustomException(e)