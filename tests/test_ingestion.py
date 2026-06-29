import asyncio
import os
from pathlib import Path
from fastapi import UploadFile
from qdrant_client import QdrantClient

# Import your modules
from ingestion.vector_store import VectorStore
from ingestion.bm25_store import BM25Store
from ingestion.metadata import MetadataStore
from services.ingestion_service import IngestionService

async def run_test():
    print("🚀 Starting Ingestion Pipeline Test...\n")

    # 1. Initialize Databases (Use local/memory versions for testing)
    print("Setting up databases...")
    qdrant_client = QdrantClient(":memory:") # Runs Qdrant in memory for testing
    vector_store = VectorStore(client=qdrant_client, collection_name="test_collection", vector_size=384)
    bm25_store = BM25Store(index_path="storage/test_bm25_index.pkl")
    metadata_store = MetadataStore(db_path="storage/test_metadata.db")

    # 2. Initialize the Ingestion Service
    service = IngestionService(
        vector_store=vector_store,
        bm25_store=bm25_store,
        metadata_store=metadata_store
    )

    # 3. Create a mock UploadFile using the sample text we created
    test_filepath = "samle_text.txt"
    if not os.path.exists(test_filepath):
        print(f"❌ Error: Please create {test_filepath} first!")
        return

    print("Uploading file to service...")
    with open(test_filepath, "rb") as f:
        # Mocking FastAPI's UploadFile
        upload_file = UploadFile(filename=test_filepath, file=f)
        
        # 4. RUN THE INGESTION PIPELINE
        try:
            result = await service.process(upload_file)
            print("\n✅ Ingestion Result:")
            print(result)
        except Exception as e:
            print(f"\n❌ Pipeline Failed: {e}")
            return

    # 5. VALIDATION: Check if data actually went into the databases
    print("\n--- Running Validations ---")
    
    # Check Metadata DB
    docs = metadata_store.list_documents()
    print(f"📄 Metadata DB Documents: {len(docs)}")
    if docs:
        doc_uuid = docs[0]['document_uuid']
        print(f"   -> Document UUID: {doc_uuid}")
        print(f"   -> Chunks Generated: {docs[0]['chunks']}")
        
    # Check BM25 Index
    bm25_data = bm25_store.load()
    if bm25_data:
        print(f"🔍 BM25 Index Documents: {len(bm25_data['documents'])}")

    # 6. (Optional) TEST DELETION FLOW
    print("\n🗑️ Testing Deletion Flow...")
    if docs:
        delete_result = service.remove_document(doc_uuid) # Using the method we added earlier
        print(delete_result)
        
        # Verify it's gone
        remaining_docs = metadata_store.list_documents()
        print(f"📄 Metadata DB Documents after deletion: {len(remaining_docs)}")

if __name__ == "__main__":
    # Ensure the storage directory exists
    Path("storage/uploads").mkdir(parents=True, exist_ok=True)
    
    # Run the async test
    asyncio.run(run_test())