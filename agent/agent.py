from . import llm
from .tools import run_tool

SYSTEM_PROMPT = """你是一个可用工具的 AI Agent。
可用工具：
- get_current_time: 获取当前时间（无参数）

若需要工具，只回复一行：TOOL:工具名
否则直接回答用户。"""


class Agent:
    def __init__(
        self,
        provider: str = "bailian",
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ):
        cfg = llm.PROVIDERS[provider]
        self.provider = provider
        self.model = model or cfg["default_model"]
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]

    def _chat(self) -> str:
        return llm.chat(
            self.messages,
            provider=self.provider,
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

    def run(self, user_input: str) -> str:
        self.messages.append({"role": "user", "content": user_input})
        try:
            reply = self._chat()

            if reply.startswith("TOOL:"):
                tool_name = reply.removeprefix("TOOL:").strip()
                result = run_tool(tool_name)
                self.messages.append({"role": "assistant", "content": reply})
                self.messages.append(
                    {"role": "user", "content": f"工具结果: {result}\n请据此回答用户。"}
                )
                reply = self._chat()

            self.messages.append({"role": "assistant", "content": reply})
            return reply
        except llm.LLMError as exc:
            self.messages.pop()
            return f"错误：{exc.message}"
