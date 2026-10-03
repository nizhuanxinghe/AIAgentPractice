import json
import re

OUTPUT_TEMPLATE = "好的，我是{role}，今年{age}岁，我会以{style}风格和你聊天。"

EXTRACT_SYSTEM = """从用户输入抽取角色设定，只输出一个 JSON 对象，不要解释。
字段：
- is_persona: 用户是否在指定身份或说话风格，true 或 false
- role: 身份，例如英语老师；没有则为空字符串
- style: 性格或风格，例如幽默；没有则为空字符串
- age: 按身份推断的整数年龄，必须是 1 到 120，不要输出 0
普通聊天时 is_persona 为 false。"""


def looks_like_persona(text: str) -> bool:
    return any(word in (text or "") for word in ("你是", "风格", "性格", "角色"))


def render_output(role: str, style: str, age: int) -> str:
    return OUTPUT_TEMPLATE.format(role=role, style=style, age=age)


def parse_slots(text: str) -> dict | None:
    raw = (text or "").strip()
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw)
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        matched = re.search(r"\{.*\}", raw, re.S)
        if not matched:
            return None
        try:
            data = json.loads(matched.group())
        except json.JSONDecodeError:
            return None
    if not isinstance(data, dict) or not data.get("is_persona"):
        return None
    role = str(data.get("role") or "").strip()
    style = str(data.get("style") or "").strip()
    try:
        age = int(data.get("age") or 0)
    except (TypeError, ValueError):
        return None
    if not role or not style:
        return None
    if not 1 <= age <= 120:
        age = 0
    return {"role": role, "style": style, "age": age}
