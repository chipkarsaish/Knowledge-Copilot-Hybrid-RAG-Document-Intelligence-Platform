from uuid import uuid4

from langchain_core.documents import Document

from qdrant_client import QdrantClient
from qdrant_client.models import (

    PointStruct,

    VectorParams,

    Distance,

    Filter,

    FieldCondition,

    MatchValue

)
from core.logger import logging
from core.exception import CustomException


class VectorStore:

    def __init__(self, client: QdrantClient, collection_name: str, vector_size: int = 768):
        self.client = client
        self.collection_name = collection_name
        self.vector_size = vector_size
        self._create_collection()


    # Create Collection
    def _create_collection(self):
        logging.info("Create a new collection")
        try:
            collections = self.client.get_collections().collections

            existing = [

                collection.name

                for collection in collections

            ]

            if self.collection_name not in existing:

                self.client.create_collection(

                    collection_name=self.collection_name,

                    vectors_config=VectorParams(

                        size=self.vector_size,

                        distance=Distance.COSINE

                    )

                )
        except Exception as e:
            raise CustomException(e)

    # Insert
    def insert(self, chunks: list[Document], embeddings: list[list[float]]):
        logging.info("Insert new vectors of the Document")
        try:
            points = []

            for chunk, embedding in zip(chunks, embeddings):

                point = PointStruct(

                    id=str(uuid4()),

                    vector=embedding,

                    payload={

                        "text": chunk.page_content,

                        **chunk.metadata

                    }

                )

                points.append(point)

            self.client.upsert(

                collection_name=self.collection_name,

                points=points

            )
        except Exception as e:
            raise CustomException(e)

    # Delete Document
    def delete(self, document_uuid: str):
        logging.info("Delete the Document related vectors")
        try:
            self.client.delete(

                collection_name=self.collection_name,

                points_selector=Filter(

                    must=[

                        FieldCondition(

                            key="document_uuid",

                            match=MatchValue(

                                value=document_uuid

                            )

                        )

                    ]

                )

            )
        except Exception as e:
            raise CustomException(e)


    # Dense Search
    def search(self, query_vector: list[float], top_k: int = 20):
        logging.info("Search the vector for a query vector")
        try:
            return self.client.query_points(

                collection_name=self.collection_name,

                query=query_vector,

                limit=top_k

            ).points
        except Exception as e:
            raise CustomException(e)