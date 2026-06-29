from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from logging import logging
from exception import CustomException

class DocumentEmbedder:

    def __init__(self, model_name: str = "ibm-granite/granite-embedding-small-english-r2"):
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={
                "device": "cpu"
            },
            encode_kwargs={
                "normalize_embeddings": True
            }
        )

    def embed(self, chunks: list[Document]) -> list[list[float]]:
        logging.info("Start Embedding")
        try: 
            texts = [
                chunk.page_content
                for chunk in chunks
            ]

            embeddings = self.embedding_model.embed_documents(texts)

            return embeddings
        except Exception as e:
            raise CustomException(e)