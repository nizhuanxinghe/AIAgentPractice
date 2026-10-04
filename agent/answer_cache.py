import json
import logging
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore

from .embeddings import get_embeddings

logger = logging.getLogger("agent")

SIMILARITY_THRESHOLD = 0.9
CACHE_PATH = Path(__file__).resolve().parent.parent / "data" / "answer_cache.json"

_cache: "AnswerCache | None" = None


class AnswerCache:
    def __init__(self, provider: str = "bailian"):
        self.store = InMemoryVectorStore(embedding=get_embeddings(provider))
        self._load()

    def lookup(self, question: str) -> tuple[str, float] | None:
        if not self.store.store:
            logger.info("向量库为空，调用模型")
            return None
        hits = self.store.similarity_search_with_score(question, k=1)
        if not hits:
            return None
        doc, score = hits[0]
        logger.info("最相近问题: %s 相似度: %.3f", doc.page_content, score)
        if score < SIMILARITY_THRESHOLD:
            return None
        return str(doc.metadata.get("answer") or ""), float(score)

    def add(self, question: str, answer: str) -> None:
        self.store.add_documents(
            [Document(page_content=question, metadata={"answer": answer})]
        )
        self._save()
        logger.info("写入向量库: %s", question)

    def _load(self) -> None:
        if not CACHE_PATH.exists():
            return
        items = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
        for item in items:
            self.store.store[item["id"]] = item
        logger.info("已加载向量库 %s 条", len(items))

    def _save(self) -> None:
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        items = []
        for item in self.store.store.values():
            saved = dict(item)
            saved["vector"] = [float(value) for value in item["vector"]]
            items.append(saved)
        CACHE_PATH.write_text(json.dumps(items, ensure_ascii=False), encoding="utf-8")


def get_answer_cache() -> AnswerCache:
    global _cache
    if _cache is None:
        _cache = AnswerCache()
    return _cache
