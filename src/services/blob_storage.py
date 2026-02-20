"""
Azure Blob Storage Connector for RAG Pipeline.
Stores uploaded PDFs and indexed document metadata.
Falls back to local filesystem when Azure is not configured.
"""

import os
import json
import uuid
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

try:
    from azure.storage.blob import BlobServiceClient, ContentSettings
    AZURE_AVAILABLE = True
except ImportError:
    AZURE_AVAILABLE = False


class BlobStorageConnector:
    """Manages document storage for the RAG pipeline."""

    def __init__(self):
        self.conn_str = os.getenv("AZURE_STORAGE_CONNECTION_STRING", "")
        self.container = os.getenv("BLOB_CONTAINER_RAG", "rag-documents")
        self.use_azure = bool(self.conn_str) and AZURE_AVAILABLE
        self.blob_service = None

        if self.use_azure:
            try:
                self.blob_service = BlobServiceClient.from_connection_string(self.conn_str)
                try:
                    self.blob_service.create_container(self.container)
                except Exception:
                    pass
                logger.info("Azure Blob Storage connected for RAG pipeline")
            except Exception as e:
                logger.warning(f"Blob init failed: {e}. Using local.")
                self.use_azure = False

        if not self.use_azure:
            os.makedirs("uploads", exist_ok=True)
            os.makedirs("outputs", exist_ok=True)

    async def store_pdf(self, file_bytes: bytes, filename: str) -> str:
        """Store uploaded PDF and return storage path."""
        blob_name = f"pdfs/{datetime.utcnow().strftime('%Y%m%d')}_{uuid.uuid4().hex[:8]}_{filename}"

        if self.use_azure:
            client = self.blob_service.get_blob_client(self.container, blob_name)
            client.upload_blob(file_bytes, overwrite=True,
                               content_settings=ContentSettings(content_type="application/pdf"))
            return f"azure://{self.container}/{blob_name}"
        else:
            path = os.path.join("uploads", blob_name.replace("/", "_"))
            with open(path, "wb") as f:
                f.write(file_bytes)
            return path

    async def store_index_metadata(self, filename: str, metadata: dict) -> str:
        """Store ingestion metadata."""
        meta_name = f"metadata/{filename}.json"
        data = json.dumps(metadata, default=str, indent=2)

        if self.use_azure:
            client = self.blob_service.get_blob_client(self.container, meta_name)
            client.upload_blob(data, overwrite=True,
                               content_settings=ContentSettings(content_type="application/json"))
            return f"azure://{self.container}/{meta_name}"
        else:
            path = os.path.join("outputs", f"{filename}_metadata.json")
            with open(path, "w") as f:
                f.write(data)
            return path
