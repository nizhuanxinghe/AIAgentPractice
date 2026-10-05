import json
import re

import requests


def normalize_city(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    if text.startswith("{"):
        try:
            data = json.loads(text)
            if isinstance(data, dict) and data.get("city") is not None:
                text = str(data["city"])
        except json.JSONDecodeError:
            pass
    match = re.search(r'city\s*[=:]\s*["\']?([^"\'}]+)', text, re.I)
    if match:
        text = match.group(1)
    text = text.strip().strip('"').strip("'").strip()
    if text.lower().startswith("city="):
        text = text[5:].strip().strip('"').strip("'")
    return text.strip()


def geocode(city: str) -> dict | None:
    response = requests.get(
        "https://geocoding-api.open-meteo.com/v1/search",
        params={"name": city, "count": 1, "language": "zh"},
        timeout=20,
    )
    response.raise_for_status()
    results = response.json().get("results") or []
    return results[0] if results else None


def format_city_info(place: dict, city: str) -> str:
    name = place.get("name") or city
    country = place.get("country") or "未知"
    admin = place.get("admin1") or "未知"
    timezone = place.get("timezone") or "未知"
    elevation = place.get("elevation")
    population = place.get("population")
    elevation_text = f"{elevation}米" if elevation is not None else "未知"
    population_text = f"{population}" if population is not None else "未知"
    return (
        f"{name}：国家/地区 {country}，省/州 {admin}，"
        f"时区 {timezone}，海拔 {elevation_text}，人口 {population_text}"
    )
