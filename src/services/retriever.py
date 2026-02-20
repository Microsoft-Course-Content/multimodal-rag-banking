"""
Hybrid Search Retriever.
Combines keyword search, vector search, and semantic ranking
to retrieve the most relevant text chunks AND images.
"""

import logging
from dataclasses import dataclass
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery
from azure.core.credentials import AzureKeyCredential

from ..config import get_settings
from .text_embedder import TextEmbedder
from .image_embedder import ImageEmbedder

logger = logging.getLogger(__name__)


@dataclass
class RetrievedChunk:
    """A retrieved chunk with relevance score."""
    id: str
    content: str
    content_type: str  # "text" or "image"
    page_number: int
    source_document: str
    section_title: str
    score: float
    image_base64: str | None = None
    image_caption: str | None = None
    has_table: bool = False


class HybridRetriever:
    """
    Retrieves relevant content using hybrid search:
    1. Keyword search (BM25) — for exact matches (numbers, codes)
    2. Vector search — for semantic similarity
    3. Semantic ranker — for reranking top results
    
    Also retrieves relevant images using cross-modal search.
    """

    def __init__(self):
        settings = get_settings()
        self.search_client = SearchClient(
            endpoint=settings.azure_search_endpoint,
            index_name=settings.azure_search_index_name,
            credential=AzureKeyCredential(settings.azure_search_api_key),
        )
        self.text_embedder = TextEmbedder()
        self.image_embedder = ImageEmbedder()
        self.top_k = settings.top_k_results
        self.similarity_threshold = settings.similarity_threshold

    async def retrieve(
        self, query: str, top_k: int | None = None, include_images: bool = True
    ) -> list[RetrievedChunk]:
        """
        Retrieve relevant text chunks and images for a query.

        Args:
            query: User's natural language question
            top_k: Number of results to return
            include_images: Whether to also search for relevant images

        Returns:
            List of RetrievedChunks ranked by relevance
        """
        k = top_k or self.top_k
        results = []

        # 1. Text hybrid search (keyword + vector)
        text_results = await self._hybrid_text_search(query, k)
        results.extend(text_results)

        # 2. Image search (cross-modal: text query → image embeddings)
        if include_images:
            image_results = await self._image_search(query, max(2, k // 2))
            results.extend(image_results)

        # 3. Deduplicate and sort by score
        seen_ids = set()
        unique_results = []
        for r in sorted(results, key=lambda x: x.score, reverse=True):
            if r.id not in seen_ids:
                seen_ids.add(r.id)
                unique_results.append(r)

        # 4. Filter by threshold
        filtered = [
            r for r in unique_results if r.score >= self.similarity_threshold
        ]

        logger.info(
            f"Retrieved {len(filtered)} chunks for query: '{query[:50]}...' "
            f"(text: {sum(1 for r in filtered if r.content_type == 'text')}, "
            f"images: {sum(1 for r in filtered if r.content_type == 'image')})"
        )
        return filtered[:k]

    async def _hybrid_text_search(
        self, query: str, top_k: int
    ) -> list[RetrievedChunk]:
        """Perform hybrid (keyword + vector) search on text chunks."""
        # Generate query embedding
        query_vector = await self.text_embedder.embed_text(query)

        vector_query = VectorizedQuery(
            vector=query_vector,
            k_nearest_neighbors=top_k,
            fields="text_vector",
        )

        results = self.search_client.search(
            search_text=query,
            vector_queries=[vector_query],
            filter="content_type eq 'text'",
            select=[
                "id", "content", "content_type", "page_number",
                "source_document", "section_title", "has_table",
            ],
            top=top_k,
        )

        chunks = []
        for result in results:
            chunks.append(RetrievedChunk(
                id=result["id"],
                content=result["content"],
                content_type="text",
                page_number=result.get("page_number", 0),
                source_document=result.get("source_document", ""),
                section_title=result.get("section_title", ""),
                score=result["@search.score"],
                has_table=result.get("has_table", False),
            ))

        return chunks

    async def _image_search(
        self, query: str, top_k: int
    ) -> list[RetrievedChunk]:
        """
        Search for relevant images using cross-modal embeddings.
        Text query → AI Vision text embedding → matches against image embeddings.
        """
        try:
            # Use AI Vision's text vectorizer (same space as image vectors)
            query_vector = await self.image_embedder.embed_text_for_image_search(query)

            vector_query = VectorizedQuery(
                vector=query_vector,
                k_nearest_neighbors=top_k,
                fields="image_vector",
            )

            results = self.search_client.search(
                search_text=None,
                vector_queries=[vector_query],
                filter="content_type eq 'image'",
                select=[
                    "id", "content", "content_type", "page_number",
                    "source_document", "image_caption", "image_base64",
                ],
                top=top_k,
            )

            chunks = []
            for result in results:
                chunks.append(RetrievedChunk(
                    id=result["id"],
                    content=result.get("content", ""),
                    content_type="image",
                    page_number=result.get("page_number", 0),
                    source_document=result.get("source_document", ""),
                    section_title="",
                    score=result["@search.score"],
                    image_base64=result.get("image_base64"),
                    image_caption=result.get("image_caption"),
                ))

            return chunks

        except Exception as e:
            logger.warning(f"Image search failed: {e}")
            return []
