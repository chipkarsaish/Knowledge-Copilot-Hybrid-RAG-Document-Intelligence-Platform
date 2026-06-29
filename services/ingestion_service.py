from fastapi import UploadFile
from ingestion.loder import DocumentLoader
from ingestion.reciever import FileReceiver
from ingestion.splitter import DocumentSplitter
from ingestion.embedder import DocumentEmbedder
from ingestion.bm25_store import BM25Store
from ingestion.vector_store import VectorStore
from ingestion.metadata import MetadataStore
import hashlib
from pathlib import Path
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
    
    def _compute_file_hash(self, file_path: Path) -> str:
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            # Read and update hash string value in blocks of 4K
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
        
    async def process(self, file: UploadFile) -> dict:
        file_path = await self.reciever.receive(file)
        
        file_hash = self._compute_file_hash(file_path)
        
        # Check if this document already exists in the metadata database
        existing_doc = self.metadata_store.get_document_by_hash(file_hash)
        if existing_doc:
            # Clean up the redundant file saved by the receiver
            Path(file_path).unlink(missing_ok=True)
            return {
                "status": "skipped",
                "filename": file.filename,
                "message": "Document already exists and was skipped to prevent duplication."
            }
        
        documents_list = self.loader.load(file_path) 
        
        document_uuid = str(uuid.uuid4())   ## generate a  UUID of the document  
        
        chunks = self.splitter.split(documents_list, doc_uuid = document_uuid)

        embeddings = self.embedder.embed(chunks)

        self.vector_store.insert(
            chunks,
            embeddings
        )

        self.bm25_store.build(chunks)
        
        # Insert the metadata, now INCLUDING the file_hash
        self.metadata_store.insert_document(
            document_uuid=document_uuid,
            filename=file.filename,
            filepath=str(file_path),
            file_hash=file_hash, 
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
        
    def remove_document(self, document_uuid: str) -> dict:
        import os
        
        # 1. Fetch metadata to get the physical filepath before we delete it
        doc_info = self.metadata_store.get_document(document_uuid)
        
        if not doc_info:
            return {"status": "error", "message": f"Document with UUID {document_uuid} not found."}

        filepath = doc_info.get("filepath")
        
        # 2. Delete the physical file from the storage/uploads directory
        if filepath and os.path.exists(filepath):
            os.remove(filepath)

        # 3. Delete dense vectors from Qdrant (VectorStore)
        self.vector_store.delete(document_uuid)

        # 4. Remove sparse keywords from the BM25 Store and rebuild index
        self.bm25_store.delete_document(document_uuid)

        # 5. Finally, remove the metadata record from SQLite
        self.metadata_store.delete_document(document_uuid)

        return {
            "status": "success",
            "document_uuid": document_uuid,
            "filename": doc_info.get("filename"),
            "message": "Document completely removed from physical storage, VectorDB, BM25, and Metadata DB."
        }
        
     
