import os

from langchain_openai import ChatOpenAI

from . import llm


def get_chat_model(
    provider: str,
    model: str | None,
    temperature: float,
) -> ChatOpenAI:
    cfg = llm.PROVIDERS[provider]
    api_key = os.getenv(cfg["api_key_env"]) or cfg.get("api_key_default")
    if not api_key:
        raise RuntimeError(f"请在 .env 中设置 {cfg['api_key_env']}")
    return ChatOpenAI(
        model=model or cfg["default_model"],
        api_key=api_key,
        base_url=cfg["base_url"],
        temperature=temperature,
    )
