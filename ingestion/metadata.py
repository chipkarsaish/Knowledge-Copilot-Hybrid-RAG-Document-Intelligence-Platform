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


class MetadataStore:

    def __init__(self, db_path: str = "storage/metadata.db"):
        self.db_path = Path(db_path)

        self.connection = sqlite3.connect(
            self.db_path,
            check_same_thread=False
        )

        self.create_table()

    def create_table(self):

        cursor = self.connection.cursor()

        cursor.execute("""

        CREATE TABLE IF NOT EXISTS documents(

            id INTEGER PRIMARY KEY AUTOINCREMENT,
            
            document_uuid TEXT UNIQUE NOT NULL,

            filename TEXT NOT NULL,

            filepath TEXT NOT NULL,

            pages INTEGER,

            chunks INTEGER,

            embedding_model TEXT,

            upload_time TEXT,

            status TEXT

        )

        """)

        self.connection.commit()

    def insert_document(self, document_uuid: str, filename: str, filepath: str, pages: int, chunks: int, embedding_model: str, status: str = "Indexed"):

        cursor = self.connection.cursor()

        cursor.execute("""

        INSERT INTO documents(document_uuid, filename, filepath, pages, chunks, embedding_model, upload_time, status)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)

        """,

        (
            document_uuid,

            filename,

            filepath,

            pages,

            chunks,

            embedding_model,

            datetime.now().isoformat(),

            status

        )

        )

        self.connection.commit()