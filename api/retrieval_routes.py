from fastapi import APIRouter, Depends, HTTPException
from schemas.api_models import ChatRequest, ChatResponse
from api.dependencies import get_retrieval_service
from services.retrieval_service import RetrievalService

router = APIRouter(prefix="/api/v1/chat", tags=["Copilot Chat"])

@router.post("/ask", response_model=ChatResponse)
def ask_question(
    request: ChatRequest, 
    service: RetrievalService = Depends(get_retrieval_service)
):
    try:
        # Passes the query to your massive RAG pipeline!
        result = service.answer_query(request.query)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))