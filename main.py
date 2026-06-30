from fastapi import FastAPI
from api import ingestion_routes, retrieval_routes
from pathlib import Path

# Ensure storage directories exist before server starts
Path("storage/uploads").mkdir(parents=True, exist_ok=True)
Path("storage/qdrant_db").mkdir(parents=True, exist_ok=True)

# Initialize FastAPI App
app = FastAPI(
    title="Knowledge Copilot API",
    description="Hybrid RAG Document Intelligence Platform",
    version="1.0.0"
)

# Plug in the Routes
app.include_router(ingestion_routes.router)
app.include_router(retrieval_routes.router)

@app.get("/", tags=["System"])
def health_check():
    return {
        "status": "Online", 
        "message": "Knowledge Copilot Engine is running!"
    }