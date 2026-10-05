import logging
import warnings

from langchain_classic.memory import ConversationBufferMemory
from langchain_core._api import LangChainDeprecationWarning

warnings.filterwarnings("ignore", category=LangChainDeprecationWarning)

from . import llm
from .answer_cache import get_answer_cache
from .documents import get_document_store
from .persona import extract_persona, persona_system
from .tool_agent import build_executor

logger = logging.getLogger("agent")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False

SYSTEM_PROMPT = "你是一个 AI Agent。直接用自然语言回答。不要提及工具名或 JSON。"


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
        self._persona: dict | None = None
        self._documents_enabled = False
        self._executor = build_executor(provider, self.model, temperature)

    def enable_documents(self) -> None:
        self._documents_enabled = True

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
        system = SYSTEM_PROMPT
        persona = persona_system(self._persona)
        if persona:
            system = f"{system}\n{persona}"
        return [{"role": "system", "content": system}, *stored]

    def remember(self, user_input: str, reply: str) -> None:
        self.memory.save_context({"input": user_input}, {"output": reply})

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

    def _update_persona(self, user_input: str) -> None:
        persona = extract_persona(user_input, self.provider, self.model)
        if persona:
            self._persona = persona
            logger.info("角色设定: %s", persona)

    def _reuse_answer(self, user_input: str) -> str | None:
        found = get_answer_cache().lookup(user_input)
        if not found:
            return None
        answer, score = found
        self.remember(user_input, answer)
        logger.info("复用答案 相似度: %.3f", score)
        self._log_output(answer)
        return answer

    def _store_answer(self, user_input: str, reply: str) -> None:
        if reply.startswith("错误"):
            return
        get_answer_cache().add(user_input, reply)

    def _document_context(self, user_input: str) -> str:
        if not self._documents_enabled:
            return ""
        return get_document_store().context(user_input)

    def _run_agent(self, user_input: str, context: str) -> str:
        question = user_input
        if context:
            question = f"{user_input}\n\n参考文档片段：\n{context}"
        result = self._executor.invoke(
            {
                "input": question,
                "persona": persona_system(self._persona) or "无额外角色设定。",
            }
        )
        return str(result.get("output") or "").strip()

    def stream(self, user_input: str):
        logger.info("输入: %s", user_input)
        try:
            self._update_persona(user_input)
            context = self._document_context(user_input)
            if not context:
                reused = self._reuse_answer(user_input)
                if reused:
                    yield reused
                    return
            shown = self._run_agent(user_input, context)
            if not shown:
                shown = "请换种说法再试一次。"
            yield shown
            self.remember(user_input, shown)
            if not context:
                self._store_answer(user_input, shown)
            self._log_output(shown)
        except Exception as exc:
            if isinstance(exc, llm.LLMError):
                reply = f"错误：{exc.message}"
            else:
                reply = f"错误：{exc}"
            self._log_output(reply)
            yield reply

    def run(self, user_input: str) -> str:
        reply = ""
        for reply in self.stream(user_input):
            pass
        return reply
