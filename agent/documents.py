import json
import logging
from pathlib import Path

from langchain_community.document_loaders import TextLoader
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .embeddings import get_embeddings

logger = logging.getLogger("agent")

CHUNK_SIZE = 500
CHUNK_OVERLAP = 80
TOP_K = 4
SCORE_FLOOR = 0.35
CHAR_LIMIT = 6000
EMBED_BATCH = 10
STORE_PATH = Path(__file__).resolve().parent.parent / "data" / "document_store.json"

_store: "DocumentStore | None" = None


class DocumentStore:
    def __init__(self, provider: str = "bailian"):
        self.store = InMemoryVectorStore(embedding=get_embeddings(provider))
        self._load()

    def ingest(self, path: str) -> int:
        docs = _read_text(path)
        chunks = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        ).split_documents(docs)
        chunks = [chunk for chunk in chunks if chunk.page_content.strip()]
        if not chunks:
            raise ValueError("文档为空")
        source = Path(path).name
        for index, chunk in enumerate(chunks):
            chunk.metadata["source"] = source
            chunk.metadata["index"] = index
        self.store.store.clear()
        for start in range(0, len(chunks), EMBED_BATCH):
            self.store.add_documents(chunks[start:start + EMBED_BATCH])
        self._save()
        logger.info("文档分块: %s 共 %s 块", source, len(chunks))
        return len(chunks)

    def context(self, query: str) -> str:
        if not self.store.store:
            return ""
        hits = self.store.similarity_search_with_score(query, k=TOP_K)
        texts: list[str] = []
        if hits and hits[0][1] >= SCORE_FLOOR:
            logger.info("文档检索 最高相似度: %.3f", hits[0][1])
            texts.extend(doc.page_content for doc, _score in hits)
        elif hits:
            logger.info("文档检索 最高相似度: %.3f，改用文首片段", hits[0][1])
        for text in self._leading():
            if text not in texts:
                texts.append(text)
        joined = _clip(texts)
        if joined:
            preview = " | ".join(
                line.split("] ", 1)[-1][:40].replace("\n", " ")
                for line in joined.splitlines()
                if line.startswith("[")
            )
            logger.info("检索片段: %s", preview)
        return joined

    def _leading(self) -> list[str]:
        items = sorted(
            self.store.store.values(),
            key=lambda item: int((item.get("metadata") or {}).get("index") or 0),
        )
        return [str(item.get("text") or "") for item in items]

    def _load(self) -> None:
        if not STORE_PATH.exists():
            return
        items = json.loads(STORE_PATH.read_text(encoding="utf-8"))
        for item in items:
            self.store.store[item["id"]] = item
        logger.info("已加载文档向量库 %s 条", len(items))

    def _save(self) -> None:
        STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
        items = []
        for item in self.store.store.values():
            saved = dict(item)
            saved["vector"] = [float(value) for value in item["vector"]]
            items.append(saved)
        STORE_PATH.write_text(json.dumps(items, ensure_ascii=False), encoding="utf-8")


def _read_text(path: str):
    last_error: Exception | None = None
    for encoding in ("utf-8", "gbk"):
        try:
            return TextLoader(path, encoding=encoding).load()
        except RuntimeError as exc:
            last_error = exc
    raise last_error or RuntimeError(f"无法读取 {path}")


def _clip(texts: list[str]) -> str:
    parts = []
    total = 0
    for index, text in enumerate(texts, 1):
        piece = text.strip()
        if not piece:
            continue
        room = CHAR_LIMIT - total
        if room <= 0:
            break
        if len(piece) > room:
            piece = piece[:room]
        parts.append(f"[{index}] {piece}")
        total += len(piece)
    return "\n".join(parts)


def get_document_store() -> DocumentStore:
    global _store
    if _store is None:
        _store = DocumentStore()
    return _store
