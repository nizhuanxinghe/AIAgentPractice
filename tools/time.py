import logging
from datetime import datetime

from langchain_core.tools import tool

logger = logging.getLogger("agent")


@tool
def get_current_time() -> str:
    """获取当前本地时间。没有参数。"""
    logger.info("工具请求: get_current_time {}")
    result = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger.info("工具返回: %s", result)
    return result
