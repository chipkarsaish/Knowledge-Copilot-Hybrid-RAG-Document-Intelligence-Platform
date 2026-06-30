from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
from typing import List
from schemas.api_models import DocumentResponse
from api.dependencies import get_ingestion_service, get_metadata_store
from services.ingestion_service import IngestionService
from ingestion.metadata import MetadataStore

router = APIRouter(prefix="/api/v1/documents", tags=["Knowledge Base Management"])

@router.post("/upload")
async def upload_document(
    file: UploadFile = File(...), 
    service: IngestionService = Depends(get_ingestion_service)
):
    try:
        # Pass the file directly to the engine you built!
        result = await service.process(file)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/", response_model=List[DocumentResponse])
def get_all_documents(store: MetadataStore = Depends(get_metadata_store)):
    try:
        # Returns all files logged in SQLite
        return store.list_documents()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/{document_uuid}")
def delete_document(
    document_uuid: str, 
    service: IngestionService = Depends(get_ingestion_service)
):
    try:
        result = service.remove_document(document_uuid)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))