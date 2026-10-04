import logging
import math
import os

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

load_dotenv()

logger = logging.getLogger("agent")

EMBEDDINGS = {
    "bailian": {
        "model": "text-embedding-v3",
        "api_key_env": "DASHSCOPE_API_KEY",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    },
    "openai": {
        "model": "text-embedding-3-small",
        "api_key_env": "OPENAI_API_KEY",
        "base_url": "https://api.openai.com/v1",
    },
    "ollama": {
        "model": "qwen2.5:3b",
        "api_key_env": "OLLAMA_API_KEY",
        "api_key_default": "ollama",
        "base_url": "http://127.0.0.1:11434/v1",
    },
}


def get_embeddings(provider: str = "bailian") -> OpenAIEmbeddings:
    cfg = EMBEDDINGS.get(provider)
    if not cfg:
        raise ValueError(f"未知 embedding provider: {provider}")
    api_key = os.getenv(cfg["api_key_env"]) or cfg.get("api_key_default")
    if not api_key:
        raise RuntimeError(f"请在 .env 中设置 {cfg['api_key_env']}")
    return OpenAIEmbeddings(
        model=cfg["model"],
        api_key=api_key,
        base_url=cfg["base_url"],
        check_embedding_ctx_length=False,
    )


def embed_query(text: str, provider: str = "bailian") -> list[float]:
    vector = get_embeddings(provider).embed_query(text)
    logger.info("嵌入维度: %s", len(vector))
    return vector


def embed_documents(texts: list[str], provider: str = "bailian") -> list[list[float]]:
    vectors = get_embeddings(provider).embed_documents(texts)
    logger.info("嵌入维度: %s", len(vectors[0]) if vectors else 0)
    return vectors


def cosine_similarity(left: list[float], right: list[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if not left_norm or not right_norm:
        return 0.0
    return dot / (left_norm * right_norm)
