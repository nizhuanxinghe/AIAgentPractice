from . import llm
from .tools import run_tool

SYSTEM_PROMPT = """你是一个可用工具的 AI Agent。
可用工具：
- get_current_time: 获取当前时间（无参数）

若需要工具，只回复一行：TOOL:工具名
否则直接回答用户。"""


class Agent:
    def __init__(self, model: str = "qwen-plus"):
        self.model = model
        self.messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]

    def run(self, user_input: str) -> str:
        self.messages.append({"role": "user", "content": user_input})
        reply = llm.chat(self.messages, model=self.model)

        if reply.startswith("TOOL:"):
            tool_name = reply.removeprefix("TOOL:").strip()
            result = run_tool(tool_name)
            self.messages.append({"role": "assistant", "content": reply})
            self.messages.append(
                {"role": "user", "content": f"工具结果: {result}\n请据此回答用户。"}
            )
            reply = llm.chat(self.messages, model=self.model)

        self.messages.append({"role": "assistant", "content": reply})
        return reply
