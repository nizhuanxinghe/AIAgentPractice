import logging

import requests
from langchain_core.tools import tool

logger = logging.getLogger("agent")

from .geocode import geocode, normalize_city

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

_CURRENT_FIELDS = (
    "temperature_2m,apparent_temperature,relative_humidity_2m,precipitation,"
    "cloud_cover,surface_pressure,wind_speed_10m,wind_direction_10m,wind_gusts_10m,"
    "weather_code"
)
_DAILY_FIELDS = (
    "temperature_2m_max,temperature_2m_min,precipitation_probability_max,"
    "uv_index_max,sunrise,sunset"
)


def _wind_direction(degrees: float | int | None) -> str:
    if degrees is None:
        return "未知"
    value = float(degrees) % 360
    dirs = ["北", "东北", "东", "东南", "南", "西南", "西", "西北"]
    index = int((value + 22.5) // 45) % 8
    return dirs[index]


def _fetch(latitude: float, longitude: float, timezone: str) -> dict:
    response = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": latitude,
            "longitude": longitude,
            "timezone": timezone,
            "current": _CURRENT_FIELDS,
            "daily": _DAILY_FIELDS,
            "forecast_days": 1,
        },
        timeout=20,
    )
    response.raise_for_status()
    return response.json()


def query(city: str) -> str:
    city = normalize_city(city)
    if not city:
        return "请说明要查询的城市"
    try:
        place = geocode(city)
        if not place:
            return f"没有找到城市：{city}"
        payload = _fetch(
            place["latitude"],
            place["longitude"],
            place.get("timezone") or "auto",
        )
    except requests.Timeout:
        return "错误：天气查询超时，请稍后再试"
    except requests.RequestException:
        return "错误：天气查询失败，请稍后再试"

    name = place.get("name") or city
    current = payload.get("current") or {}
    daily = payload.get("daily") or {}
    code = int(current.get("weather_code", -1))
    sky = _WEATHER_TEXT.get(code, "未知")
    lines = [
        f"{name} 此刻 {sky}",
        f"气温 {current.get('temperature_2m')}°C，体感 {current.get('apparent_temperature')}°C",
        f"湿度 {current.get('relative_humidity_2m')}%，降水量 {current.get('precipitation')}mm",
        f"云量 {current.get('cloud_cover')}%，气压 {current.get('surface_pressure')}hPa",
        (
            f"风速 {current.get('wind_speed_10m')}km/h，"
            f"风向 {_wind_direction(current.get('wind_direction_10m'))}，"
            f"阵风 {current.get('wind_gusts_10m')}km/h"
        ),
    ]
    if daily.get("time"):
        lines.append(
            "当天 "
            f"最高 {daily.get('temperature_2m_max', [None])[0]}°C，"
            f"最低 {daily.get('temperature_2m_min', [None])[0]}°C，"
            f"降水概率 {daily.get('precipitation_probability_max', [None])[0]}%，"
            f"紫外线 {daily.get('uv_index_max', [None])[0]}"
        )
        sunrise = (daily.get("sunrise") or [None])[0]
        sunset = (daily.get("sunset") or [None])[0]
        if sunrise or sunset:
            lines.append(f"日出 {sunrise or '未知'}，日落 {sunset or '未知'}")
    return "\n".join(lines)


@tool
def get_weather(city: str) -> str:
    """查询某个城市的完整天气，含此刻与当天预报。city 是城市名，例如北京。"""
    logger.info("工具请求: get_weather %s", {"city": city})
    result = query(city)
    logger.info("工具返回: %s", result)
    return result
