from pydantic import BaseModel
from typing import List, Optional

# --- Requests ---
class ChatRequest(BaseModel):
    query: str

# --- Responses ---
class CitationResponse(BaseModel):
    filename: str
    page: int
    cross_encoder_score: float

class ChatResponse(BaseModel):
    question: str
    answer: str
    confidence_score: float
    citations: List[CitationResponse]

class DocumentResponse(BaseModel):
    document_uuid: str
    filename: str
    upload_time: str
    status: str