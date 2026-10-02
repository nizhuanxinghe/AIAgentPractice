import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


def get_client() -> OpenAI:
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise RuntimeError("请在 .env 中设置 DASHSCOPE_API_KEY")
    return OpenAI(
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )


def chat(messages: list[dict], model: str = "qwen-plus") -> str:
    client = get_client()
    response = client.chat.completions.create(model=model, messages=messages)
    return response.choices[0].message.content or ""
