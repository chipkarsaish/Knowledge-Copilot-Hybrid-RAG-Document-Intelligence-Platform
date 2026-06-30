from pathlib import Path
import pickle

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi
from core.logger import logging
from core.exception import CustomException


class BM25Store:

    def __init__(self, index_path: str = "storage/bm25_index.pkl"):
        self.index_path = Path(index_path)

    def build(self,chunks: list[Document]):
        logging.info("Build the bm25 storage for Documents")
        try:
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
        except Exception as e:
            raise CustomException(e)
        
    # ---------------------------------------------------
    # Load Index
    # ---------------------------------------------------

    def load(self):
        logging.info("Loading BM25 index")

        try:
            if not self.index_path.exists():
                logging.warning("BM25 index file does not exist.")
                return None

            with open(self.index_path, "rb") as f:
                data = pickle.load(f)

            logging.info("BM25 index loaded successfully.")
            return data

        except Exception as e:
            raise CustomException(e)


    # ---------------------------------------------------
    # Search
    # ---------------------------------------------------

    def search(
        self,
        query: str,
        top_k: int = 20
    ) -> list[Document]:

        logging.info(f"Searching BM25 index for query: '{query}'")

        try:
            data = self.load()

            if data is None:
                logging.warning("BM25 index is empty or not found.")
                return []

            bm25 = data["bm25"]
            documents = data["documents"]

            query_tokens = query.split()

            scores = bm25.get_scores(query_tokens)

            ranked = sorted(
                enumerate(scores),
                key=lambda x: x[1],
                reverse=True
            )

            results = []

            for idx, score in ranked[:top_k]:
                doc = documents[idx]
                doc.metadata["bm25_score"] = float(score)
                results.append(doc)

            logging.info(
                f"Retrieved {len(results)} document(s) for the query."
            )

            return results

        except Exception as e:
            raise CustomException(e)


    # ---------------------------------------------------
    # Delete Index
    # ---------------------------------------------------

    def delete_index(self):
        logging.info("Deleting BM25 index")

        try:
            if self.index_path.exists():
                self.index_path.unlink()
                logging.info("BM25 index deleted successfully.")
            else:
                logging.warning("BM25 index file does not exist.")

        except Exception as e:
            raise CustomException(e)
        
    def delete_document(self, document_uuid: str):
        logging.info(f"Removing document {document_uuid} from BM25 index")
        try:
            data = self.load()
            if data is None:
                return

            documents = data["documents"]
            
            # Filter out all chunks that belong to the target document_uuid
            remaining_documents = [
                doc for doc in documents 
                if doc.metadata.get("document_uuid") != document_uuid
            ]

            # If the lengths are the same, the document wasn't in here
            if len(remaining_documents) == len(documents):
                logging.info("Document not found in BM25 index.")
                return

            # Rebuild the BM25 index with the remaining documents
            if len(remaining_documents) > 0:
                self.build(remaining_documents)
            else:
                # If no documents are left, just delete the whole index file
                self.delete_index()

        except Exception as e:
            raise CustomException(e)