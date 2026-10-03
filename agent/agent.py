import re

from . import llm
from .chain import step, stop
from .prompt import EXTRACT_SYSTEM, looks_like_persona, parse_slots, render_output
from .tools import run_tool

SYSTEM_PROMPT = """你是一个可用工具的 AI Agent。
可用工具：
- get_current_time: 获取当前时间（无参数）

若需要工具，只回复一行：TOOL:工具名
否则直接回答用户。"""


def _call(ctx, messages, temperature, max_tokens):
    agent = ctx["agent"]
    return llm.chat(
        messages,
        provider=agent.provider,
        model=agent.model,
        temperature=temperature,
        max_tokens=max_tokens,
    )


def extract_persona(ctx):
    ctx["raw"] = _call(
        ctx,
        [
            {"role": "system", "content": EXTRACT_SYSTEM},
            {"role": "user", "content": ctx["text"]},
        ],
        temperature=0,
        max_tokens=128,
    )
    return ctx


def parse_persona(ctx):
    persona = parse_slots(ctx["raw"])
    if not persona:
        return stop(None)
    ctx["persona"] = persona
    return ctx


def fill_age(ctx):
    persona = ctx["persona"]
    if persona["age"]:
        return ctx
    reply = _call(
        ctx,
        [
            {"role": "system", "content": "根据身份推断一个合理整数年龄，只输出数字。"},
            {"role": "user", "content": persona["role"]},
        ],
        temperature=0,
        max_tokens=16,
    )
    match = re.search(r"\d{1,3}", reply)
    if not match:
        raise llm.LLMError("无法根据角色推断年龄")
    persona["age"] = int(match.group())
    return ctx


def render_persona(ctx):
    persona = ctx["persona"]
    reply = render_output(persona["role"], persona["style"], persona["age"])
    agent = ctx["agent"]
    agent.messages.append({"role": "user", "content": ctx["text"]})
    agent.messages.append({"role": "assistant", "content": reply})
    return reply


persona_chain = step(extract_persona) | parse_persona | fill_age | render_persona


def route_persona(ctx):
    if not looks_like_persona(ctx["text"]):
        return ctx
    reply = persona_chain.invoke(ctx)
    if reply:
        return stop(reply)
    return ctx


def add_user(ctx):
    ctx["agent"].messages.append({"role": "user", "content": ctx["text"]})
    return ctx


def complete(ctx):
    agent = ctx["agent"]
    ctx["reply"] = agent._chat()
    return ctx


def apply_tool(ctx):
    reply = ctx["reply"]
    if not reply.startswith("TOOL:"):
        return ctx
    agent = ctx["agent"]
    tool_name = reply.removeprefix("TOOL:").strip()
    result = run_tool(tool_name)
    agent.messages.append({"role": "assistant", "content": reply})
    agent.messages.append({"role": "user", "content": f"工具结果: {result}\n请据此回答用户。"})
    ctx["reply"] = agent._chat()
    return ctx


def save_reply(ctx):
    reply = ctx["reply"]
    ctx["agent"].messages.append({"role": "assistant", "content": reply})
    return reply


chat_chain = step(add_user) | complete | apply_tool | save_reply
agent_chain = step(route_persona) | chat_chain


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
        try:
            return agent_chain.invoke({"agent": self, "text": user_input})
        except llm.LLMError as exc:
            if self.messages and self.messages[-1]["role"] == "user":
                self.messages.pop()
            return f"错误：{exc.message}"
