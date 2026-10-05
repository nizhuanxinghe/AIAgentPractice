import json
import re

from . import llm

PERSONA_SCHEMA = {
    "type": "object",
    "properties": {
        "is_persona": {
            "type": "boolean",
            "description": "用户是否在指定助手身份或说话风格",
        },
        "role": {"type": "string", "description": "身份，例如英语老师"},
        "style": {"type": "string", "description": "性格或风格，例如幽默"},
        "age": {
            "type": "integer",
            "description": "按身份推断的合理年龄，1 到 120",
        },
    },
    "required": ["is_persona", "role", "style", "age"],
}

EXTRACT_SYSTEM = (
    "判断用户是否在设定助手身份或说话风格。只输出一个 JSON 对象，不要解释。\n"
    f"JSON Schema：{json.dumps(PERSONA_SCHEMA, ensure_ascii=False)}\n"
    "普通聊天时 is_persona 为 false，role/style 为空字符串，age 为 0。"
)


def _parse_json(raw: str) -> dict | None:
    text = (raw or "").strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.S)
    if fenced:
        text = fenced.group(1)
    else:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            return None
        text = text[start : end + 1]
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def extract_persona(user_input: str, provider: str, model: str) -> dict | None:
    raw = llm.chat(
        [
            {"role": "system", "content": EXTRACT_SYSTEM},
            {"role": "user", "content": user_input},
        ],
        provider=provider,
        model=model,
        temperature=0,
        max_tokens=256,
    )
    data = _parse_json(raw)
    if not data or not data.get("is_persona"):
        return None
    role = str(data.get("role") or "").strip()
    style = str(data.get("style") or "").strip()
    if not role or not style:
        return None
    try:
        age = int(data.get("age") or 0)
    except (TypeError, ValueError):
        age = 0
    if not 1 <= age <= 120:
        age = 0
    return {"role": role, "style": style, "age": age}


def persona_system(persona: dict | None) -> str:
    if not persona:
        return ""
    age = persona.get("age") or 0
    age_text = f"，今年{age}岁" if age else ""
    return f"你是{persona['role']}{age_text}，以{persona['style']}风格和用户对话。"
