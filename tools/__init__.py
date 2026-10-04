from .time import get_current_time
from .weather import get_weather

TOOLS = [get_current_time, get_weather]
TOOL_MAP = {item.name: item for item in TOOLS}

__all__ = ["TOOLS", "TOOL_MAP", "get_current_time", "get_weather"]
