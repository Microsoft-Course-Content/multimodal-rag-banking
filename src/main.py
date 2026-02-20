"""
Multimodal RAG Pipeline for Banking — FastAPI Application.
Upload financial reports with charts and ask questions in natural language.

Author: Jalal Ahmed Khan

Run locally:   uvicorn src.main:app --reload --port 8001
Run on Azure:  Deployed as Azure App Service (see README)
"""

import logging
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from .models.schemas import QueryRequest, QueryResponse, IngestResponse
from .services.rag_pipeline import RAGPipeline
from .services.index_manager import IndexManager
from .services.blob_storage import BlobStorageConnector

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = BASE_DIR / "static"

pipeline = RAGPipeline()
index_manager = IndexManager()
blob_storage = BlobStorageConnector()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing Multimodal RAG Pipeline...")
    await index_manager.create_index()
    logger.info("Ready — accepting requests")
    yield
    logger.info("Shutting down...")


app = FastAPI(
    title="Multimodal RAG Pipeline — Banking",
    description=(
        "Upload banking PDFs with charts and tables, then ask questions. "
        "Uses Azure OpenAI GPT-4o, Azure AI Search (hybrid), and "
        "Azure AI Vision image embeddings for multimodal retrieval."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Serve static files
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", include_in_schema=False)
async def serve_frontend():
    """Serve the Web UI."""
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    return {"service": "Multimodal RAG Pipeline", "docs": "/docs"}


@app.post("/api/v1/ingest", response_model=IngestResponse)
async def ingest_document(file: UploadFile = File(...)):
    """Ingest a PDF document into the RAG index."""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are supported")

    file_bytes = await file.read()
    if len(file_bytes) > 100 * 1024 * 1024:
        raise HTTPException(400, "File too large (max 100MB)")

    # Store PDF in blob storage
    storage_path = await blob_storage.store_pdf(file_bytes, file.filename)
    logger.info(f"PDF stored: {storage_path}")

    # Ingest into RAG pipeline
    result = await pipeline.ingest(file_bytes, file.filename)

    # Store metadata
    await blob_storage.store_index_metadata(file.filename, result)

    return IngestResponse(**result)


@app.post("/api/v1/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """Ask a question about ingested banking documents."""
    result = await pipeline.query(request.query, top_k=request.top_k)
    return QueryResponse(**result)


@app.delete("/api/v1/documents/{filename}")
async def delete_document(filename: str):
    """Remove a document from the index."""
    await index_manager.delete_by_source(filename)
    return {"status": "deleted", "filename": filename}


@app.get("/api/v1/health")
async def health():
    return {"status": "healthy", "service": "Multimodal RAG Pipeline"}


@app.get("/")
async def root():
    return {
        "service": "Multimodal RAG Pipeline — Banking",
        "version": "1.0.0",
        "docs": "/docs",
    }
