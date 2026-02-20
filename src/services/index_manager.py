"""
Azure AI Search Index Manager.
Creates and manages the hybrid search index with text vectors,
image vectors, and metadata for the multimodal RAG pipeline.
"""

import logging
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex,
    SearchField,
    SearchFieldDataType,
    SimpleField,
    SearchableField,
    VectorSearch,
    HnswAlgorithmConfiguration,
    VectorSearchProfile,
    SearchIndex,
)
from azure.core.credentials import AzureKeyCredential
from ..config import get_settings

logger = logging.getLogger(__name__)


class IndexManager:
    """Manages Azure AI Search index for multimodal RAG."""

    def __init__(self):
        settings = get_settings()
        credential = AzureKeyCredential(settings.azure_search_api_key)
        self.index_client = SearchIndexClient(
            endpoint=settings.azure_search_endpoint,
            credential=credential,
        )
        self.search_client = SearchClient(
            endpoint=settings.azure_search_endpoint,
            index_name=settings.azure_search_index_name,
            credential=credential,
        )
        self.index_name = settings.azure_search_index_name

    async def create_index(self):
        """Create the multimodal search index with vector fields."""
        fields = [
            SimpleField(
                name="id", type=SearchFieldDataType.String,
                key=True, filterable=True
            ),
            SearchableField(
                name="content", type=SearchFieldDataType.String,
                analyzer_name="en.microsoft"
            ),
            SimpleField(
                name="content_type", type=SearchFieldDataType.String,
                filterable=True  # "text" or "image"
            ),
            SimpleField(
                name="page_number", type=SearchFieldDataType.Int32,
                filterable=True, sortable=True
            ),
            SimpleField(
                name="source_document", type=SearchFieldDataType.String,
                filterable=True
            ),
            SimpleField(
                name="section_title", type=SearchFieldDataType.String,
                filterable=True
            ),
            SimpleField(
                name="has_table", type=SearchFieldDataType.Boolean,
                filterable=True
            ),
            SimpleField(
                name="image_caption", type=SearchFieldDataType.String,
            ),
            SimpleField(
                name="image_base64", type=SearchFieldDataType.String,
            ),
            SearchField(
                name="text_vector",
                type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                searchable=True,
                vector_search_dimensions=1536,
                vector_search_profile_name="text-vector-profile",
            ),
            SearchField(
                name="image_vector",
                type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                searchable=True,
                vector_search_dimensions=1024,
                vector_search_profile_name="image-vector-profile",
            ),
        ]

        vector_search = VectorSearch(
            algorithms=[
                HnswAlgorithmConfiguration(name="text-hnsw", parameters={"m": 4, "efConstruction": 400, "efSearch": 500}),
                HnswAlgorithmConfiguration(name="image-hnsw", parameters={"m": 4, "efConstruction": 400, "efSearch": 500}),
            ],
            profiles=[
                VectorSearchProfile(name="text-vector-profile", algorithm_configuration_name="text-hnsw"),
                VectorSearchProfile(name="image-vector-profile", algorithm_configuration_name="image-hnsw"),
            ],
        )

        index = SearchIndex(
            name=self.index_name,
            fields=fields,
            vector_search=vector_search,
        )

        self.index_client.create_or_update_index(index)
        logger.info(f"Index '{self.index_name}' created/updated")

    async def upload_documents(self, documents: list[dict]):
        """Upload documents to the search index."""
        batch_size = 100
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            result = self.search_client.upload_documents(batch)
            succeeded = sum(1 for r in result if r.succeeded)
            logger.info(f"Uploaded batch: {succeeded}/{len(batch)} succeeded")

    async def delete_by_source(self, source_document: str):
        """Delete all chunks from a specific source document."""
        results = self.search_client.search(
            search_text="*",
            filter=f"source_document eq '{source_document}'",
            select=["id"],
        )
        doc_ids = [{"id": r["id"]} for r in results]
        if doc_ids:
            self.search_client.delete_documents(doc_ids)
            logger.info(f"Deleted {len(doc_ids)} chunks for {source_document}")
