"""
Image Embedding Service using Azure AI Vision (Florence model).
Generates vector embeddings for document images (charts, graphs, figures)
to enable image-based retrieval in the RAG pipeline.
"""

import base64
import logging
import httpx
from ..config import get_settings

logger = logging.getLogger(__name__)


class ImageEmbedder:
    """
    Generates image embeddings using Azure AI Vision's
    vectorize-image endpoint (Florence foundation model).
    
    These embeddings exist in the same vector space as text embeddings
    from the vectorize-text endpoint, enabling cross-modal search.
    """

    def __init__(self):
        settings = get_settings()
        self.endpoint = settings.azure_vision_endpoint.rstrip("/")
        self.api_key = settings.azure_vision_api_key
        self.api_version = "2024-02-01"
        self.dimension = 1024  # Florence model output dimension

    async def embed_image(self, image_bytes: bytes) -> list[float]:
        """
        Generate embedding for a single image.

        Args:
            image_bytes: Raw image bytes (PNG, JPEG, etc.)

        Returns:
            1024-dimensional float vector
        """
        url = (
            f"{self.endpoint}/computervision/retrieval:vectorizeImage"
            f"?api-version={self.api_version}&model-version=2023-04-15"
        )

        headers = {
            "Ocp-Apim-Subscription-Key": self.api_key,
            "Content-Type": "application/octet-stream",
        }

        async with httpx.AsyncClient() as client:
            response = await client.post(
                url, headers=headers, content=image_bytes, timeout=30.0
            )
            response.raise_for_status()
            result = response.json()

        embedding = result["vector"]
        logger.debug(f"Generated image embedding: {len(embedding)} dimensions")
        return embedding

    async def embed_image_base64(self, base64_image: str) -> list[float]:
        """Generate embedding from base64-encoded image."""
        image_bytes = base64.b64decode(base64_image)
        return await self.embed_image(image_bytes)

    async def embed_text_for_image_search(self, text: str) -> list[float]:
        """
        Generate text embedding in the IMAGE vector space.
        Uses Azure AI Vision's vectorize-text endpoint so that
        text queries can match against image embeddings.
        
        This is different from the OpenAI text embeddings — these
        live in the same space as image embeddings (CLIP-like).
        """
        url = (
            f"{self.endpoint}/computervision/retrieval:vectorizeText"
            f"?api-version={self.api_version}&model-version=2023-04-15"
        )

        headers = {
            "Ocp-Apim-Subscription-Key": self.api_key,
            "Content-Type": "application/json",
        }

        payload = {"text": text}

        async with httpx.AsyncClient() as client:
            response = await client.post(
                url, headers=headers, json=payload, timeout=30.0
            )
            response.raise_for_status()
            result = response.json()

        return result["vector"]

    async def embed_batch(self, images: list[bytes]) -> list[list[float]]:
        """Generate embeddings for multiple images sequentially."""
        embeddings = []
        for idx, image_bytes in enumerate(images):
            try:
                embedding = await self.embed_image(image_bytes)
                embeddings.append(embedding)
            except Exception as e:
                logger.warning(f"Failed to embed image {idx}: {e}")
                embeddings.append([0.0] * self.dimension)  # Zero vector fallback

        logger.info(f"Generated {len(embeddings)} image embeddings")
        return embeddings
