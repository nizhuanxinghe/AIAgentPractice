from datetime import datetime


def get_current_time() -> str:
    """返回当前本地时间。"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


TOOLS = {
    "get_current_time": get_current_time,
}


def run_tool(name: str, **kwargs) -> str:
    fn = TOOLS.get(name)
    if not fn:
        return f"未知工具: {name}"
    return str(fn(**kwargs))
