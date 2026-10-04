from datetime import datetime

from langchain_core.tools import tool


@tool
def get_current_time() -> str:
    """获取当前本地时间。没有参数。"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
