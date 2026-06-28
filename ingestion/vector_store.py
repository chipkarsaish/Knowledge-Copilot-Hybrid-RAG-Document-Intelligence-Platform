from uuid import uuid4

from langchain_core.documents import Document

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct


class VectorStore:

    def __init__(self, collection_name: str = "knowledge_base", vector_size: int = 768):

        self.collection_name = collection_name

        self.client = QdrantClient(
            host="localhost",
            port=6333
        )

        collections = [
            c.name
            for c in self.client.get_collections().collections
        ]

        if self.collection_name not in collections:

            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=Distance.COSINE
                )
            )

    def insert(self, chunks: list[Document], embeddings: list[list[float]]) -> None:
        """
        Insert document chunks into Qdrant.
        """

        points = []

        for chunk, embedding in zip(chunks, embeddings):

            payload = {
                "text": chunk.page_content,
                **chunk.metadata
            }

            points.append(

                PointStruct(

                    id=str(uuid4()),

                    vector=embedding,

                    payload=payload
                )
            )

        self.client.upsert(

            collection_name=self.collection_name,

            points=points
        )