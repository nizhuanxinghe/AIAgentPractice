import logging

logger = logging.getLogger("agent")

_disabled = False
_warned = False


def embeddings_enabled() -> bool:
    return not _disabled


def disable_embeddings() -> None:
    global _disabled, _warned
    _disabled = True
    if not _warned:
        logger.info("嵌入服务不可用，已跳过文档检索与答案复用")
        _warned = True


def note_embedding_error(exc: Exception) -> None:
    text = str(exc)
    if "Arrearage" in text or "insufficient_quota" in text or "invalid_api_key" in text:
        disable_embeddings()
        return
    if "400" in text and "Access denied" in text:
        disable_embeddings()
