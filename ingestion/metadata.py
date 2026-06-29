'''
Remember, Qdrant already stores chunk-level metadata:
Chunk
Page
Source
Chunk ID
## We need to store the File related info in this Database

Schema: TAble(documents)
| Column          | Type    | Purpose                          |
| --------------- | ------- | -------------------------------- |
| id              | INTEGER | Primary Key                      |
| document_uuid   | STRING  | Unique document identifier       |
| filename        | TEXT    | Uploaded filename                |
| filepath        | TEXT    | Where the file is stored         |
| pages           | INTEGER | Number of pages/documents loaded |
| chunks          | INTEGER | Number of chunks generated       |
| embedding_model | TEXT    | Which embedding model indexed it |
| upload_time     | TEXT    | Upload timestamp                 |
| status          | TEXT    | Indexed / Failed / Processing    |

'''
from pathlib import Path
import sqlite3
from datetime import datetime
from dataclasses import dataclass
from logging import logging
from exception import CustomException


class MetadataStore:

    def __init__(self, db_path: str = "storage/metadata.db"):
        self.db_path = Path(db_path)
        self.connection = sqlite3.connect(
            self.db_path,
            check_same_thread=False
        )
        self.connection.row_factory = sqlite3.Row
        self.create_table()

    def create_table(self):
        logging.info("Create Meta data TAble")
        try:

            cursor = self.connection.cursor()

            cursor.execute("""

            CREATE TABLE IF NOT EXISTS documents(

                id INTEGER PRIMARY KEY AUTOINCREMENT,
                
                document_uuid TEXT UNIQUE NOT NULL,

                filename TEXT NOT NULL,

                filepath TEXT NOT NULL,
                
                file_hash TEXT UNIQUE NOT NULL,

                pages INTEGER,

                chunks INTEGER,

                embedding_model TEXT,

                upload_time TEXT,

                status TEXT

            )

            """)

            self.connection.commit()
        except Exception as e:
            raise CustomException(e)

    def insert_document(self, document_uuid: str, filename: str, filepath: str, file_hash: str, pages: int, chunks: int, embedding_model: str, status: str = "Indexed"):
        logging.info("Insert the Document info in the meta-database")
        try:
            cursor = self.connection.cursor()

            cursor.execute("""

            INSERT INTO documents(document_uuid, filename, filepath, file_hash, pages, chunks, embedding_model, upload_time, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)

            """,

            (
                document_uuid,

                filename,

                filepath,
                
                file_hash,

                pages,

                chunks,

                embedding_model,

                datetime.now().isoformat(),

                status

            )

            )

            self.connection.commit()
        except Exception as e:
            raise CustomException(e)
        

    # --------------------------------------------------
    # Get by UUID
    # --------------------------------------------------

    def get_document(self, document_uuid: str):
        logging.info("Get the document info")
        try:
            cursor = self.connection.cursor()

            cursor.execute(

                """

                SELECT *

                FROM documents

                WHERE document_uuid = ?

                """,

                (document_uuid,)

            )

            row = cursor.fetchone()

            return dict(row) if row else None
        
        except Exception as e:
            raise CustomException(e)

    # --------------------------------------------------
    # Get by filename
    # --------------------------------------------------

    def get_document_by_filename(self, filename: str):
        logging.info("Get Document info by file name")
        try:
            cursor = self.connection.cursor()

            cursor.execute(

                """

                SELECT *

                FROM documents

                WHERE filename = ?

                """,

                (filename,)

            )

            row = cursor.fetchone()

            return dict(row) if row else None
        except Exception as e:
            raise CustomException(e)

    # --------------------------------------------------
    # Get by Hash
    # --------------------------------------------------

    def get_document_by_hash(self, file_hash: str):
        logging.info("Get Document by Hash of Document")
        try: 
            cursor = self.connection.cursor()

            cursor.execute(

                """

                SELECT *

                FROM documents

                WHERE file_hash = ?

                """,

                (file_hash,)

            )

            row = cursor.fetchone()

            return dict(row) if row else None
        except Exception as e:
            raise CustomException(e)

    # --------------------------------------------------
    # List Documents
    # --------------------------------------------------

    def list_documents(self):
        logging.info("Listing all indexed documents")
        try:
            cursor = self.connection.cursor()

            cursor.execute(
                """
                SELECT *
                FROM documents
                ORDER BY upload_time DESC
                """
            )

            rows = cursor.fetchall()

            return [dict(row) for row in rows]

        except Exception as e:
            raise CustomException(e)

    # --------------------------------------------------
    # Delete
    # --------------------------------------------------

    def delete_document(self, document_uuid: str):
        logging.info(f"Deleting document: {document_uuid}")

        try:
            cursor = self.connection.cursor()

            cursor.execute(
                """
                DELETE FROM documents
                WHERE document_uuid = ?
                """,
                (document_uuid,)
            )

            self.connection.commit()

            logging.info(f"Document deleted successfully: {document_uuid}")

        except Exception as e:
            raise CustomException(e)

    # --------------------------------------------------
    # Update Status
    # --------------------------------------------------

    def update_status(
        self,
        document_uuid: str,
        status: str
    ):
        logging.info(
            f"Updating status of document {document_uuid} to '{status}'"
        )

        try:
            cursor = self.connection.cursor()

            cursor.execute(
                """
                UPDATE documents
                SET status = ?
                WHERE document_uuid = ?
                """,
                (
                    status,
                    document_uuid
                )
            )

            self.connection.commit()

            logging.info(
                f"Status updated successfully for document: {document_uuid}"
            )

        except Exception as e:
            raise CustomException(e)

    # --------------------------------------------------
    # Close
    # --------------------------------------------------

    def close(self):
        logging.info("Closing metadata database connection")

        try:
            self.connection.close()
            logging.info("Metadata database connection closed successfully")

        except Exception as e:
            raise CustomException(e)


@dataclass
class DocumentMetadata:
    document_uuid: str
    filename: str
    filepath: str
    file_hash: str
    pages: int
    chunks: int
    embedding_model: str
    upload_time: str
    status: str