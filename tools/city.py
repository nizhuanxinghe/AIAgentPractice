import logging

import requests
from langchain_core.tools import tool

logger = logging.getLogger("agent")

from .geocode import format_city_info, geocode, normalize_city


def query_city(city: str) -> str:
    city = normalize_city(city)
    if not city:
        return "请说明要查询的城市"
    try:
        place = geocode(city)
    except requests.Timeout:
        return "错误：城市查询超时，请稍后再试"
    except requests.RequestException:
        return "错误：城市查询失败，请稍后再试"
    if not place:
        return f"没有找到城市：{city}"
    return format_city_info(place, city)


@tool
def get_city_info(city: str) -> str:
    """查询城市的国家、省或州、时区、海拔、人口。city 是城市名，例如北京。"""
    logger.info("工具请求: get_city_info %s", {"city": city})
    result = query_city(city)
    logger.info("工具返回: %s", result)
    return result
