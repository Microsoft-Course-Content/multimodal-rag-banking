"""Pydantic models for Multimodal RAG Pipeline API."""

from pydantic import BaseModel, Field
from typing import Optional


class QueryRequest(BaseModel):
    query: str = Field(..., description="Natural language question")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of chunks to retrieve")
    include_images: bool = Field(default=True, description="Include image search results")


class Citation(BaseModel):
    page: int
    source: str
    section: str = ""
    content_type: str = "text"


class QueryResponse(BaseModel):
    answer: str
    citations: list[Citation] = []
    chunks_used: int = 0
    images_used: int = 0
    processing_time_seconds: float = 0.0
    tokens_used: Optional[dict] = None


class IngestResponse(BaseModel):
    filename: str
    pages_processed: int
    text_chunks: int
    images_indexed: int
    total_documents_indexed: int
    processing_time_seconds: float
