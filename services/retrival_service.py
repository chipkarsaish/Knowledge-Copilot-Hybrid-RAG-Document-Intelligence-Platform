from core.logger import logging
import json
from core.exception import CustomException
from retrieval.dense import DenseRetriever
from retrieval.sparse import SparseRetriever
from retrieval.reranker import RRFReranker
from retrieval.cross_encoder import EntropyReranker

from langchain_core.prompts import PromptTemplate

class RetrievalService:
    def __init__(
        self, 
        dense_retriever: DenseRetriever, 
        sparse_retriever: SparseRetriever, 
        rrf_reranker: RRFReranker,
        entropy_reranker: EntropyReranker,
        llm
    ):
        self.dense = dense_retriever
        self.sparse = sparse_retriever
        self.rrf = rrf_reranker
        self.entropy = entropy_reranker
        self.llm = llm

        # We ask the LLM to return JSON so we can extract Confidence and Citations
        self.prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""You are the Knowledge Copilot, an AI assistant. Answer the question based ONLY on the following context. 
            
            Instructions:
            1. Answer the question accurately.
            2. If you don't know the answer, say "I don't have enough information."
            3. Provide a confidence score between 0.0 and 1.0.
            4. List the Source IDs of the chunks you actually used to form your answer.
            
            Respond EXACTLY in this JSON format:
            {{
                "answer": "your final answer here",
                "confidence_score": 0.95,
                "used_source_ids": ["source_1", "source_2"]
            }}

Context:
{context}

Question: {question}

JSON Response:"""
        )

    def answer_query(self, query: str) -> dict:
        logging.info(f"Processing query pipeline for: {query}")
        try:
            # 1. DENSE & SPARSE RETRIEVAL (Fetch top 20 each for a wide net)
            dense_docs = self.dense.retrieve(query, top_k=20)
            sparse_docs = self.sparse.retrieve(query, top_k=20)

            # 2. RRF FUSION (Instantly merge and get the top 15 candidates)
            rrf_docs = self.rrf.rerank(dense_docs, sparse_docs, top_n=15)

            # 3. CROSS-ENCODER RERANKER (Deep read to find the absolute Top 5)
            final_top_5 = self.entropy.rerank(query, rrf_docs, top_n=5)

            # 4. PROMPT BUILDER
            context_string = ""
            for i, doc in enumerate(final_top_5):
                # We label each chunk with an ID so the LLM can cite it
                source_id = f"source_{i+1}"
                doc["source_id"] = source_id 
                context_string += f"--- [Source ID: {source_id}] ---\n{doc['text']}\n\n"

            final_prompt = self.prompt_template.format(
                context=context_string, 
                question=query
            )

            # 5. GENERATE WITH LLM
            llm_response = self.llm.invoke(final_prompt)
            response_text = llm_response.content if hasattr(llm_response, "content") else str(llm_response)
            
            # Clean up the JSON if the LLM wrapped it in markdown code blocks
            if response_text.startswith("```json"):
                response_text = response_text.strip("```json").strip("```").strip()

            # 6. CITATION VERIFICATION & CONFIDENCE SCORE
            # Parse the LLM's JSON output
            try:
                structured_output = json.loads(response_text)
            except json.JSONDecodeError:
                # Fallback if the LLM failed to output valid JSON
                structured_output = {
                    "answer": response_text,
                    "confidence_score": 0.0,
                    "used_source_ids": []
                }

            # Map the used source IDs back to the actual document metadata
            verified_citations = []
            for doc in final_top_5:
                if doc["source_id"] in structured_output.get("used_source_ids", []):
                    verified_citations.append({
                        "filename": doc["metadata"].get("filename", "Unknown"),
                        "page": doc["metadata"].get("page", 0),
                        "cross_encoder_score": doc.get("cross_encoder_score")
                    })

            # 7. FINAL ANSWER
            return {
                "question": query,
                "answer": structured_output.get("answer", ""),
                "confidence_score": structured_output.get("confidence_score", 0.0),
                "citations": verified_citations
            }

        except Exception as e:
            raise CustomException(e)