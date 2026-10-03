import os

from dotenv import load_dotenv
from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI, OpenAIError

load_dotenv()

PROVIDERS = {
    "bailian": {
        "api_key_env": "DASHSCOPE_API_KEY",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "default_model": "qwen-plus",
    },
    "openai": {
        "api_key_env": "OPENAI_API_KEY",
        "base_url": "https://api.openai.com/v1",
        "default_model": "gpt-4o-mini",
    },
    "ollama": {
        "api_key_env": "OLLAMA_API_KEY",
        "api_key_default": "ollama",
        "base_url": "http://127.0.0.1:11434/v1",
        "default_model": "qwen2.5:3b",
    },
}

MODEL_CHOICES = {
    "百炼 / qwen-plus": ("bailian", "qwen-plus"),
    "百炼 / qwen-max": ("bailian", "qwen-max"),
    "OpenAI / gpt-4o-mini": ("openai", "gpt-4o-mini"),
    "OpenAI / gpt-4o": ("openai", "gpt-4o"),
    "Ollama / qwen2.5:3b": ("ollama", "qwen2.5:3b"),
}

STATUS_MESSAGES = {
    400: "请求参数有误，请检查输入内容。",
    401: "API Key 无效或未授权，请检查 .env 配置。",
    403: "没有访问权限，请确认账号权限或模型开通状态。",
    404: "模型或接口不存在，请更换模型后重试。",
    408: "请求超时，请稍后重试。",
    429: "请求过于频繁或额度不足，请稍后重试或检查账单额度。",
    500: "服务端内部错误，请稍后重试。",
    502: "网关错误，请稍后重试。",
    503: "服务暂时不可用，请稍后重试。",
}

CODE_MESSAGES = {
    "insufficient_quota": "账户额度不足，请前往对应平台充值后再试。",
    "credit_balance_exhausted": "账户余额已耗尽，请充值后重试。",
    "rate_limit_exceeded": "触发限流，请稍后再试。",
    "invalid_api_key": "API Key 无效，请检查 .env 配置。",
    "model_not_found": "所选模型不可用，请更换模型。",
    "context_length_exceeded": "上下文过长，请清空对话后重试。",
}


class LLMError(Exception):
    def __init__(self, message: str, status_code: int | None = None, code: str | None = None):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.code = code


def map_api_error(exc: Exception) -> LLMError:
    if isinstance(exc, APITimeoutError):
        return LLMError(STATUS_MESSAGES[408], status_code=408)
    if isinstance(exc, APIConnectionError):
        return LLMError("网络连接失败，请检查网络后重试。")

    if isinstance(exc, APIStatusError):
        status = getattr(exc, "status_code", None)
        code = None
        detail = None
        try:
            err = (exc.response.json().get("error") or {})
            code = err.get("code")
            detail = err.get("message")
        except Exception:
            pass

        if code in CODE_MESSAGES:
            msg = CODE_MESSAGES[code]
        elif status in STATUS_MESSAGES:
            msg = STATUS_MESSAGES[status]
        else:
            msg = f"调用失败（HTTP {status}）。"
            if detail:
                msg = f"{msg}（{detail}）"
        return LLMError(msg, status_code=status, code=code)

    if isinstance(exc, OpenAIError):
        return LLMError(f"模型调用失败：{exc}")
    return LLMError(f"未知错误：{exc}")


def get_client(provider: str = "bailian") -> OpenAI:
    cfg = PROVIDERS.get(provider)
    if not cfg:
        raise ValueError(f"未知 provider: {provider}")
    api_key = os.getenv(cfg["api_key_env"]) or cfg.get("api_key_default")
    if not api_key:
        raise RuntimeError(f"请在 .env 中设置 {cfg['api_key_env']}")
    return OpenAI(api_key=api_key, base_url=cfg["base_url"])


def chat(
    messages: list[dict],
    provider: str = "bailian",
    model: str | None = None,
    temperature: float = 0.5,
    max_tokens: int = 1024,
) -> str:
    cfg = PROVIDERS[provider]
    client = get_client(provider)
    try:
        response = client.chat.completions.create(
            model=model or cfg["default_model"],
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content or ""
    except Exception as exc:
        raise map_api_error(exc) from exc
