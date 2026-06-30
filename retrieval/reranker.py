from core.logger import logging
from core.exception import CustomException

class RRFReranker:
    def __init__(self, k: int = 60):
        # k is a constant used in RRF (60 is the industry standard)
        self.k = k

    def rerank(self, dense_results: list[dict], sparse_results: list[dict], top_n: int = 5) -> list[dict]:
        logging.info("Applying Reciprocal Rank Fusion (RRF) to combine results")
        try:
            # We use the text as a unique identifier to merge results
            rrf_scores = {}
            combined_docs = {}

            # Process Dense Results
            for rank, doc in enumerate(dense_results):
                text = doc["text"]
                combined_docs[text] = doc
                rrf_scores[text] = rrf_scores.get(text, 0.0) + (1.0 / (self.k + rank + 1))

            # Process Sparse Results
            for rank, doc in enumerate(sparse_results):
                text = doc["text"]
                if text not in combined_docs:
                    combined_docs[text] = doc
                rrf_scores[text] = rrf_scores.get(text, 0.0) + (1.0 / (self.k + rank + 1))

            # Sort by RRF score descending
            ranked_texts = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)

            # Return the top N documents
            final_results = []
            for text in ranked_texts[:top_n]:
                doc = combined_docs[text]
                doc["rrf_score"] = rrf_scores[text]
                final_results.append(doc)

            return final_results
        except Exception as e:
            raise CustomException(e)