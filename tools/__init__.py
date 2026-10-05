from .city import get_city_info
from .time import get_current_time
from .weather import get_weather

TOOLS = [get_current_time, get_weather, get_city_info]
TOOL_MAP = {item.name: item for item in TOOLS}

__all__ = ["TOOLS", "TOOL_MAP", "get_city_info", "get_current_time", "get_weather"]
