import requests
from langchain_core.tools import tool

_WEATHER_TEXT = {
    0: "晴",
    1: "大部晴朗",
    2: "多云",
    3: "阴",
    45: "雾",
    48: "雾凇",
    51: "小毛毛雨",
    53: "毛毛雨",
    55: "大毛毛雨",
    56: "冻毛毛雨",
    57: "强冻毛毛雨",
    61: "小雨",
    63: "中雨",
    65: "大雨",
    66: "冻雨",
    67: "强冻雨",
    71: "小雪",
    73: "中雪",
    75: "大雪",
    77: "雪粒",
    80: "小阵雨",
    81: "阵雨",
    82: "强阵雨",
    85: "小阵雪",
    86: "阵雪",
    95: "雷暴",
    96: "雷暴伴小冰雹",
    99: "雷暴伴大冰雹",
}


def _geocode(city: str) -> dict | None:
    response = requests.get(
        "https://geocoding-api.open-meteo.com/v1/search",
        params={"name": city, "count": 1, "language": "zh"},
        timeout=20,
    )
    response.raise_for_status()
    results = response.json().get("results") or []
    return results[0] if results else None


def _current(latitude: float, longitude: float) -> dict:
    response = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": latitude,
            "longitude": longitude,
            "current": "temperature_2m,weather_code",
        },
        timeout=20,
    )
    response.raise_for_status()
    return response.json().get("current") or {}


def query(city: str) -> str:
    city = (city or "").strip()
    if not city:
        return "请说明要查询的城市"
    try:
        place = _geocode(city)
        if not place:
            return f"没有找到城市：{city}"
        current = _current(place["latitude"], place["longitude"])
    except requests.Timeout:
        return "错误：天气查询超时，请稍后再试"
    except requests.RequestException:
        return "错误：天气查询失败，请稍后再试"
    code = int(current.get("weather_code", -1))
    sky = _WEATHER_TEXT.get(code, "未知")
    name = place.get("name") or city
    return f"{name}当前{sky}，气温{current.get('temperature_2m')}°C"


@tool
def get_weather(city: str) -> str:
    """查询某个城市的当前天气。city 是城市名，例如北京。"""
    return query(city)
