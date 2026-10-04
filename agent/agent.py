import logging
import re
import warnings

from langchain_classic.memory import ConversationBufferMemory
from langchain_core._api import LangChainDeprecationWarning

warnings.filterwarnings("ignore", category=LangChainDeprecationWarning)

from . import llm
from .answer_cache import get_answer_cache
from .documents import get_document_store
from .chain import step, stop
from .prompt import EXTRACT_SYSTEM, looks_like_persona, parse_slots, render_output
from tools.weather import WeatherTool
from .tools import run_tool

logger = logging.getLogger("agent")
_weather = WeatherTool()
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False

SYSTEM_PROMPT = """你是一个可用工具的 AI Agent。
可用工具：
- get_current_time: 获取当前时间（无参数）

只有用户明确询问当前时间或日期时，才只回复一行：TOOL:get_current_time
其他问题直接回答。禁止在回答中出现 TOOL、工具名或调用过程。"""

_TOOL_CALL = re.compile(r"^\s*TOOL\s*[:：]\s*([A-Za-z0-9_]+)\s*$", re.I)
_TOOL_MENTION = re.compile(r"TOOL\s*[:：]\s*[A-Za-z0-9_]+", re.I)
_TIME_QUERY = re.compile(r"几点|时间|日期|今天几号|现在是")


def _tool_name(text: str) -> str | None:
    match = _TOOL_CALL.match((text or "").strip())
    return match.group(1) if match else None


def _pending_tool(text: str) -> bool:
    stripped = (text or "").strip()
    if not stripped or _tool_name(stripped):
        return True
    return any(prefix.startswith(stripped) for prefix in ("TOOL:", "TOOL：", "tool:", "tool："))


def _visible_text(text: str) -> str:
    kept = []
    for line in (text or "").splitlines():
        if _tool_name(line) or _TOOL_MENTION.search(line):
            cleaned = _TOOL_MENTION.sub("", line).strip(" ，,。")
            if cleaned:
                kept.append(cleaned)
            continue
        if line.strip():
            kept.append(line.strip())
    return "\n".join(kept).strip()


def _asks_time(text: str) -> bool:
    return bool(_TIME_QUERY.search(text or ""))


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
    agent.remember(ctx["text"], reply)
    return reply


persona_chain = step(extract_persona) | parse_persona | fill_age | render_persona


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
        self.memory = ConversationBufferMemory(return_messages=True)
        self._turn: list[dict] = []

    @property
    def messages(self) -> list[dict]:
        history = self.memory.load_memory_variables({}).get("history") or []
        stored = [
            {
                "role": "user" if msg.type == "human" else "assistant",
                "content": msg.content,
            }
            for msg in history
        ]
        return [{"role": "system", "content": SYSTEM_PROMPT}, *stored, *self._turn]

    def remember(self, user_input: str, reply: str) -> None:
        self.memory.save_context({"input": user_input}, {"output": reply})
        self._turn.clear()

    def _log_memory(self) -> None:
        history = self.memory.load_memory_variables({}).get("history") or []
        lines = [
            f"{'用户' if msg.type == 'human' else '助手'}: {msg.content}"
            for msg in history
        ]
        logger.info("记忆:\n%s", "\n".join(lines) or "(空)")

    def _log_output(self, reply: str) -> None:
        logger.info("输出: %s", reply)
        self._log_memory()

    def _stream_chat(self):
        yield from llm.stream(
            self.messages,
            provider=self.provider,
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

    def _stream_visible(self):
        text = ""
        for delta in self._stream_chat():
            text += delta
            if _pending_tool(text):
                continue
            shown = _visible_text(text)
            if shown:
                yield shown
        return text

    def _reuse_answer(self, user_input: str) -> str | None:
        if _asks_time(user_input) or _weather.matches(user_input):
            return None
        try:
            found = get_answer_cache().lookup(user_input)
        except Exception as exc:
            logger.info("向量检索失败: %s", exc)
            return None
        if not found:
            return None
        answer, score = found
        self.remember(user_input, answer)
        logger.info("复用答案 相似度: %.3f", score)
        self._log_output(answer)
        return answer

    def _store_answer(self, user_input: str, reply: str) -> None:
        if _asks_time(user_input) or _weather.matches(user_input) or reply.startswith("错误"):
            return
        try:
            get_answer_cache().add(user_input, reply)
        except Exception as exc:
            logger.info("写入向量库失败: %s", exc)

    def _document_context(self, user_input: str) -> str:
        try:
            return get_document_store().context(user_input)
        except Exception as exc:
            logger.info("文档检索失败: %s", exc)
            return ""

    def stream(self, user_input: str):
        logger.info("输入: %s", user_input)
        try:
            if _weather.matches(user_input):
                reply = _weather.answer(user_input)
                self.remember(user_input, reply)
                self._log_output(reply)
                yield reply
                return
            context = self._document_context(user_input)
            if not context:
                reused = self._reuse_answer(user_input)
                if reused:
                    yield reused
                    return
            if not context and looks_like_persona(user_input):
                reply = persona_chain.invoke({"agent": self, "text": user_input})
                if reply:
                    self._store_answer(user_input, reply)
                    self._log_output(reply)
                    yield reply
                    return
            prompt = user_input
            if context:
                prompt = f"{user_input}\n\n参考文档片段：\n{context}"
            self._turn.append({"role": "user", "content": prompt})
            text = yield from self._stream_visible()
            tool = _tool_name(text)
            if tool == "get_current_time" and _asks_time(user_input):
                result = run_tool(tool)
                self._turn.append(
                    {
                        "role": "user",
                        "content": f"当前时间是 {result}。请直接回答用户，不要提及工具或 TOOL。",
                    }
                )
                text = yield from self._stream_visible()
            elif tool:
                self._turn.append(
                    {"role": "user", "content": "不要调用任何工具，直接用自然语言回答。"}
                )
                text = yield from self._stream_visible()
            shown = _visible_text(text) or "请换种说法再试一次。"
            yield shown
            self.remember(user_input, shown)
            if not context:
                self._store_answer(user_input, shown)
            self._log_output(shown)
        except llm.LLMError as exc:
            self._turn.clear()
            reply = f"错误：{exc.message}"
            self._log_output(reply)
            yield reply

    def run(self, user_input: str) -> str:
        reply = ""
        for reply in self.stream(user_input):
            pass
        return reply
