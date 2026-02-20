"""
End-to-End Multimodal RAG Pipeline Orchestrator.
Coordinates ingestion (crack → chunk → embed → index) and
query (embed → retrieve → generate) workflows.
"""

import base64
import logging
import time
from .document_cracker import DocumentCracker
from .chunker import SemanticChunker
from .text_embedder import TextEmbedder
from .image_embedder import ImageEmbedder
from .index_manager import IndexManager
from .retriever import HybridRetriever
from .generator import MultimodalGenerator

logger = logging.getLogger(__name__)


class RAGPipeline:
    """Orchestrates the full multimodal RAG pipeline."""

    def __init__(self):
        self.cracker = DocumentCracker()
        self.chunker = SemanticChunker()
        self.text_embedder = TextEmbedder()
        self.image_embedder = ImageEmbedder()
        self.index_manager = IndexManager()
        self.retriever = HybridRetriever()
        self.generator = MultimodalGenerator()

    async def ingest(self, file_bytes: bytes, filename: str) -> dict:
        """
        Full ingestion pipeline: PDF → chunks → embeddings → index.

        Args:
            file_bytes: Raw PDF bytes
            filename: Original filename

        Returns:
            Ingestion summary with stats
        """
        start_time = time.time()

        # Step 1: Crack the document
        logger.info(f"[Ingest] Step 1: Cracking {filename}")
        cracked = await self.cracker.crack(file_bytes, filename)

        # Step 2: Chunk text content
        logger.info(f"[Ingest] Step 2: Chunking text")
        chunks = self.chunker.chunk_document(cracked.pages, filename)

        # Step 3: Generate text embeddings
        logger.info(f"[Ingest] Step 3: Generating text embeddings")
        texts = [c.content for c in chunks]
        text_embeddings = await self.text_embedder.embed_batch(texts)

        # Step 4: Generate image embeddings
        logger.info(f"[Ingest] Step 4: Generating image embeddings")
        all_images = []
        for page in cracked.pages:
            for img in page.images:
                all_images.append(img)

        image_embeddings = []
        if all_images:
            image_bytes_list = [img["bytes"] for img in all_images]
            image_embeddings = await self.image_embedder.embed_batch(image_bytes_list)

        # Step 5: Prepare documents for indexing
        logger.info(f"[Ingest] Step 5: Indexing documents")
        search_docs = []

        # Index text chunks
        for chunk, embedding in zip(chunks, text_embeddings):
            search_docs.append({
                "id": chunk.chunk_id,
                "content": chunk.content,
                "content_type": "text",
                "page_number": chunk.page_number,
                "source_document": filename,
                "section_title": chunk.section_title,
                "has_table": chunk.has_table,
                "image_caption": "",
                "image_base64": "",
                "text_vector": embedding,
                "image_vector": [0.0] * 1024,  # Empty image vector for text
            })

        # Index images
        for img, embedding in zip(all_images, image_embeddings):
            img_b64 = base64.b64encode(img["bytes"]).decode("utf-8")
            img_id = f"{filename}_img_p{img['page_number']}_{img['image_index']}"
            search_docs.append({
                "id": img_id,
                "content": img.get("caption", f"Image from page {img['page_number']}"),
                "content_type": "image",
                "page_number": img["page_number"],
                "source_document": filename,
                "section_title": "",
                "has_table": False,
                "image_caption": img.get("caption", ""),
                "image_base64": img_b64,
                "text_vector": [0.0] * 1536,  # Empty text vector for images
                "image_vector": embedding,
            })

        # Upload to index
        await self.index_manager.upload_documents(search_docs)

        elapsed = time.time() - start_time
        summary = {
            "filename": filename,
            "pages_processed": cracked.total_pages,
            "text_chunks": len(chunks),
            "images_indexed": len(all_images),
            "total_documents_indexed": len(search_docs),
            "processing_time_seconds": round(elapsed, 2),
        }

        logger.info(f"[Ingest] Complete: {summary}")
        return summary

    async def query(self, question: str, top_k: int = 5) -> dict:
        """
        Full query pipeline: question → retrieve → generate answer.

        Args:
            question: User's natural language question
            top_k: Number of chunks to retrieve

        Returns:
            Generated answer with citations and metadata
        """
        start_time = time.time()

        # Step 1: Retrieve relevant chunks (text + images)
        logger.info(f"[Query] Retrieving for: '{question[:60]}...'")
        chunks = await self.retriever.retrieve(question, top_k=top_k)

        if not chunks:
            return {
                "answer": "I couldn't find relevant information in the indexed documents to answer this question.",
                "citations": [],
                "chunks_used": 0,
                "processing_time_seconds": round(time.time() - start_time, 2),
            }

        # Step 2: Generate answer using GPT-4o multimodal
        logger.info(f"[Query] Generating answer from {len(chunks)} chunks")
        result = await self.generator.generate(question, chunks)

        result["processing_time_seconds"] = round(time.time() - start_time, 2)
        return result
