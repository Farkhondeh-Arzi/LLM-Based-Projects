import random
from datetime import datetime, timedelta


# ---------------------------------------------------------------------
# Part 1: the real tools functions (simulated)
# ---------------------------------------------------------------------

def get_weather(city: str) -> dict:
    conditions = ["آفتابی", "ابری", "بارانی", "برفی", "غبارآلود"]
    temp = random.randint(-5, 40)
    condition = random.choice(conditions)
    return {
        "city": city,
        "temperature_celsius": temp,
        "condition": condition,
    }


def calculate(expression: str) -> dict:
    import ast
    import operator

    allowed_ops = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
    }

    def safe_eval(node):
        if isinstance(node, ast.Constant):
            return node.value
        elif isinstance(node, ast.BinOp):
            op_type = type(node.op)
            if op_type not in allowed_ops:
                raise ValueError(f"عملگر مجاز نیست: {op_type}")
            return allowed_ops[op_type](safe_eval(node.left), safe_eval(node.right))
        elif isinstance(node, ast.UnaryOp):
            op_type = type(node.op)
            if op_type not in allowed_ops:
                raise ValueError(f"عملگر مجاز نیست: {op_type}")
            return allowed_ops[op_type](safe_eval(node.operand))
        else:
            raise ValueError(f"عبارت مجاز نیست: {node}")

    try:
        tree = ast.parse(expression, mode="eval")
        result = safe_eval(tree.body)
        return {"expression": expression, "result": result}
    except Exception as e:
        return {"expression": expression, "error": str(e)}


def get_calendar_events(date: str) -> dict:
    fake_events_db = {
        "today": ["جلسه تیم ساعت ۱۰ صبح", "ددلاین پروژه ساعت ۶ عصر"],
        "tomorrow": ["مصاحبه کاری ساعت ۲ بعدازظهر"],
    }
    events = fake_events_db.get(date, [])
    return {"date": date, "events": events if events else ["رویدادی ثبت نشده"]}


# ---------------------------------------------------------------------
# Part 2: Tool name to real function mapping (to call in agent_loop.py)
# ---------------------------------------------------------------------

AVAILABLE_TOOLS = {
    "get_weather": get_weather,
    "calculate": calculate,
    "get_calendar_events": get_calendar_events,
}


# ---------------------------------------------------------------------
# Part 3: Schema for every tool -- explanation for the model
# ---------------------------------------------------------------------

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "گرفتن اطلاعات آب‌وهوای فعلی برای یک شهر مشخص.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "اسم شهر (مثلاً تهران، اصفهان)",
                    },
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": (
                "انجام یک محاسبه ریاضی دقیق (جمع، تفریق، ضرب، تقسیم، توان). "
                "برای هر سوال ریاضی، حتماً از این ابزار استفاده کن، خودت "
                "حدس نزن چون ممکنه اشتباه محاسبه کنی."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "عبارت ریاضی به فرمت پایتون، مثلاً '23 * 47' یا '(10 + 5) / 3'",
                    },
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_calendar_events",
            "description": "گرفتن لیست رویدادهای تقویم برای یک روز مشخص.",
            "parameters": {
                "type": "object",
                "properties": {
                    "date": {
                        "type": "string",
                        "description": "تاریخ مورد نظر. فقط مقادیر 'today' یا 'tomorrow' پشتیبانی می‌شود.",
                        "enum": ["today", "tomorrow"],
                    },
                },
                "required": ["date"],
            },
        },
    },
]


# ---------------------------------------------------------------------
# Part 4: Convert schema to Gemini format
# ---------------------------------------------------------------------

def get_gemini_tool_schemas() -> list[dict]:
    """تبدیل TOOL_SCHEMAS (فرمت OpenAI-style) به فرمت مسطح Gemini."""
    gemini_schemas = []
    for schema in TOOL_SCHEMAS:
        func = schema["function"]
        gemini_schemas.append({
            "type": "function",
            "name": func["name"],
            "description": func["description"],
            "parameters": func["parameters"],
        })
    return gemini_schemas


if __name__ == "__main__":
    # تست سریع هر ابزار به‌صورت مستقل، بدون درگیر کردن مدل
    print("تست get_weather:")
    print(get_weather("تهران"))

    print("\nتست calculate:")
    print(calculate("23 * 47 + 10"))
    print(calculate("100 / 0"))  # باید خطا رو gracefully هندل کنه

    print("\nتست get_calendar_events:")
    print(get_calendar_events("today"))