"""
Multimodal Answer Generator using Azure OpenAI GPT-4o.
Generates grounded answers using both text context and images.
"""

import logging
from openai import AzureOpenAI
from ..config import get_settings
from .retriever import RetrievedChunk

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a banking document analysis expert. Answer the user's question 
based ONLY on the provided context (text excerpts and images from financial documents).

Rules:
1. Only use information from the provided context
2. If the context doesn't contain enough information, say so clearly
3. When referencing data, cite the page number: [Page X]
4. When interpreting a chart or image, describe what you observe
5. For financial figures, be precise with numbers and currencies
6. If multiple sources conflict, note the discrepancy

Format your response clearly with the answer first, then supporting details."""


class MultimodalGenerator:
    """
    Generates answers using GPT-4o's multimodal capabilities.
    Combines text context and images in a single prompt for 
    grounded, accurate responses about banking documents.
    """

    def __init__(self):
        settings = get_settings()
        self.client = AzureOpenAI(
            azure_endpoint=settings.azure_openai_endpoint,
            api_key=settings.azure_openai_api_key,
            api_version=settings.azure_openai_api_version,
        )
        self.deployment = settings.azure_openai_deployment_name

    async def generate(
        self, query: str, retrieved_chunks: list[RetrievedChunk]
    ) -> dict:
        """
        Generate a grounded answer using text and image context.

        Args:
            query: User's question
            retrieved_chunks: Retrieved text chunks and images

        Returns:
            Dict with answer, citations, and metadata
        """
        # Build multimodal message content
        content_parts = self._build_context(query, retrieved_chunks)

        response = self.client.chat.completions.create(
            model=self.deployment,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": content_parts},
            ],
            max_tokens=1500,
            temperature=0.2,  # Low temperature for factual accuracy
        )

        answer = response.choices[0].message.content

        # Extract citations from the answer
        citations = self._extract_citations(answer, retrieved_chunks)

        return {
            "answer": answer,
            "citations": citations,
            "chunks_used": len(retrieved_chunks),
            "images_used": sum(1 for c in retrieved_chunks if c.content_type == "image"),
            "model": self.deployment,
            "tokens_used": {
                "prompt": response.usage.prompt_tokens,
                "completion": response.usage.completion_tokens,
                "total": response.usage.total_tokens,
            },
        }

    def _build_context(
        self, query: str, chunks: list[RetrievedChunk]
    ) -> list[dict]:
        """
        Build multimodal message content with text and images.
        GPT-4o can process both in a single message.
        """
        content = []

        # Add text context
        text_context = "CONTEXT FROM BANKING DOCUMENTS:\n\n"
        for chunk in chunks:
            if chunk.content_type == "text":
                source_info = f"[Page {chunk.page_number}, {chunk.source_document}]"
                section = f" — {chunk.section_title}" if chunk.section_title else ""
                text_context += f"{source_info}{section}:\n{chunk.content}\n\n"

        content.append({"type": "text", "text": text_context})

        # Add images
        for chunk in chunks:
            if chunk.content_type == "image" and chunk.image_base64:
                caption = chunk.image_caption or f"Image from page {chunk.page_number}"
                content.append({
                    "type": "text",
                    "text": f"\n[Image from Page {chunk.page_number}: {caption}]",
                })
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{chunk.image_base64}",
                        "detail": "high",
                    },
                })

        # Add the user's question
        content.append({
            "type": "text",
            "text": f"\nQUESTION: {query}\n\nProvide a detailed answer based on the context above.",
        })

        return content

    def _extract_citations(
        self, answer: str, chunks: list[RetrievedChunk]
    ) -> list[dict]:
        """Extract page citations from the generated answer."""
        citations = []
        seen_pages = set()

        for chunk in chunks:
            page = chunk.page_number
            if page not in seen_pages and f"Page {page}" in answer:
                seen_pages.add(page)
                citations.append({
                    "page": page,
                    "source": chunk.source_document,
                    "section": chunk.section_title,
                    "content_type": chunk.content_type,
                })

        return citations
