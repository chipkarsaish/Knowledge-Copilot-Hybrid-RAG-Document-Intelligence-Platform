from fastapi import UploadFile
from .loder import DocumentLoader
from .reciever import FileReceiver
from .splitter import DocumentSplitter
from .embedder import DocumentEmbedder
from .bm25_store import BM25Store
from .vector_store import VectorStore
from .metadata import MetadataStore
import uuid

class IngestionService:
    def __init__(self, vector_store: VectorStore, bm25_store: BM25Store, metadata_store: MetadataStore):
        self.reciever = FileReceiver()
        self.loader = DocumentLoader()
        self.splitter = DocumentSplitter()
        self.embedder = DocumentEmbedder()
        self.bm25_store = bm25_store
        self.vector_store = vector_store
        self.metadata_store = metadata_store
        
    async def process(self, file: UploadFile) -> dict:
        file_path = await self.reciever.receive(file)
        
        documents_list = self.loader.load(file_path) 
        
        document_uuid = str(uuid.uuid4())   ## generate a  UUID of the document  
        
        chunks = self.splitter.split(documents_list, doc_uuid = document_uuid)

        embeddings = self.embedder.embed(chunks)

        self.vector_store.insert(
            chunks,
            embeddings
        )

        self.bm25_store.build(chunks)

        self.metadata_store.insert_document(
            
            document_uuid=document_uuid,

            filename=file.filename,

            filepath=str(file_path),

            pages=len(documents_list),

            chunks=len(chunks),

            embedding_model="BAAI/bge-base-en-v1.5",

            status="Indexed"

        )

        return {

            "status": "success",

            "filename": file.filename,

            "pages": len(documents_list),

            "chunks": len(chunks),

            "message": "Document indexed successfully."

        }
        
     
