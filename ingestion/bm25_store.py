from pathlib import Path
import pickle

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi


class BM25Store:

    def __init__(self, index_path: str = "storage/bm25_index.pkl"):
        self.index_path = Path(index_path)

    def build(self,chunks: list[Document]):
        tokenized_documents = [

            chunk.page_content.split()

            for chunk in chunks

        ]

        bm25 = BM25Okapi(tokenized_documents)

        with open(self.index_path, "wb") as f:
            pickle.dump(
                {

                    "bm25": bm25,

                    "documents": chunks

                },

                f

            )