"""
Text Embedding Service using Azure OpenAI text-embedding-ada-002.
Generates vector embeddings for text chunks for semantic search.
"""

import logging
from openai import AzureOpenAI
from ..config import get_settings

logger = logging.getLogger(__name__)


class TextEmbedder:
    """Generates text embeddings using Azure OpenAI."""

    def __init__(self):
        settings = get_settings()
        self.client = AzureOpenAI(
            azure_endpoint=settings.azure_openai_endpoint,
            api_key=settings.azure_openai_api_key,
            api_version=settings.azure_openai_api_version,
        )
        self.deployment = settings.azure_openai_embedding_deployment
        self.dimension = 1536  # ada-002 output dimension

    async def embed_text(self, text: str) -> list[float]:
        """Generate embedding for a single text."""
        response = self.client.embeddings.create(
            input=text,
            model=self.deployment,
        )
        return response.data[0].embedding

    async def embed_batch(self, texts: list[str], batch_size: int = 16) -> list[list[float]]:
        """
        Generate embeddings for a batch of texts.
        Azure OpenAI supports up to 16 texts per request.
        """
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            response = self.client.embeddings.create(
                input=batch,
                model=self.deployment,
            )
            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)

            logger.debug(f"Embedded batch {i // batch_size + 1}: {len(batch)} texts")

        logger.info(f"Generated {len(all_embeddings)} text embeddings")
        return all_embeddings
