"""Configuration for Multimodal RAG Pipeline."""

from pydantic_settings import BaseSettings
from pydantic import Field
from functools import lru_cache


class Settings(BaseSettings):
    # Azure OpenAI
    azure_openai_endpoint: str = Field(...)
    azure_openai_api_key: str = Field(...)
    azure_openai_deployment_name: str = Field(default="gpt-4o")
    azure_openai_embedding_deployment: str = Field(default="text-embedding-ada-002")
    azure_openai_api_version: str = Field(default="2024-12-01-preview")

    # Azure AI Search
    azure_search_endpoint: str = Field(...)
    azure_search_api_key: str = Field(...)
    azure_search_index_name: str = Field(default="banking-multimodal-rag")

    # Azure AI Vision (for image embeddings)
    azure_vision_endpoint: str = Field(...)
    azure_vision_api_key: str = Field(...)

    # RAG Settings
    chunk_size: int = Field(default=512)
    chunk_overlap: int = Field(default=100)
    top_k_results: int = Field(default=5)
    similarity_threshold: float = Field(default=0.7)
    reranking_enabled: bool = Field(default=True)

    class Config:
        env_file = ".env"


@lru_cache()
def get_settings() -> Settings:
    return Settings()
