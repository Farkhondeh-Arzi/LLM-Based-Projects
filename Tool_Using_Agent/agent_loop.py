import os
import json
from google import genai

from tools import AVAILABLE_TOOLS, get_gemini_tool_schemas


SYSTEM_INSTRUCTION = """تو یک دستیار هوشمند هستی که به ابزارهای زیر دسترسی داری:
آب‌وهوا، ماشین‌حساب، و تقویم.

قوانین مهم:
۱. هروقت برای جواب دادن به یک سوال به اطلاعات دقیق نیاز داری (آب‌وهوا،
   محاسبه ریاضی، رویدادهای تقویم)، حتماً از ابزار مربوطه استفاده کن.
   خودت حدس نزن یا اطلاعات نساز.
۲. بعد از گرفتن نتیجه‌ی ابزار، بر اساس همون نتیجه به زبان فارسی و
   واضح به کاربر جواب بده.
۳. اگه سوال نیاز به ابزار نداره (مثلاً سلام‌واحوال‌پرسی)، مستقیم و
   بدون ابزار جواب بده.
"""


def execute_tool_call(tool_name: str, arguments: dict) -> dict:
    if tool_name not in AVAILABLE_TOOLS:
        return {"error": f"ابزار '{tool_name}' وجود ندارد یا مجاز نیست."}

    func = AVAILABLE_TOOLS[tool_name]
    try:
        return func(**arguments)
    except Exception as e:
        return {"error": f"خطا در اجرای ابزار: {str(e)}"}


class GeminiToolAgent:

    def __init__(
        self,
        model_name: str = "gemini-3.8-flash",
        api_key: str | None = None,
        max_iterations: int = 5,
    ):
        self.model_name = model_name
        self.max_iterations = max_iterations
        self.client = genai.Client(api_key=api_key or os.environ.get("GEMINI_API_KEY"))
        self.tools = get_gemini_tool_schemas()

    def run(self, user_question: str, verbose: bool = True) -> str:
        history = [
            {
                "type": "user_input",
                "content": [{"type": "text", "text": user_question}],
            }
        ]

        for iteration in range(self.max_iterations):
            if verbose:
                print(f"\n--- دور {iteration + 1} ---")

            interaction = self.client.interactions.create(
                model=self.model_name,
                store=False,
                input=history,
                tools=self.tools,
                system_instruction=SYSTEM_INSTRUCTION,
            )

            # stateless
            for step in interaction.steps:
                history.append(step.model_dump())

            function_call_steps = [s for s in interaction.steps if s.type == "function_call"]

            # mode 1: model calls tool
            if function_call_steps:
                for call_step in function_call_steps:
                    tool_name = call_step.name
                    arguments = call_step.arguments

                    if verbose:
                        print(f"مدل درخواست ابزار داد: {tool_name}({arguments})")

                    result = execute_tool_call(tool_name, arguments)

                    if verbose:
                        print(f"نتیجه ابزار: {result}")

                    history.append({
                        "type": "function_result",
                        "name": tool_name,
                        "call_id": call_step.id,
                        "result": [{"type": "text", "text": json.dumps(result, ensure_ascii=False)}],
                    })

                continue

            # mode 2: model returns text
            if verbose:
                print("مدل به جواب نهایی رسید.")
            return interaction.output_text

        return "متاسفانه بعد از چند تلاش نتونستم به یک جواب نهایی برسم."


if __name__ == "__main__":
    agent = GeminiToolAgent(model_name="gemini-3.8-flash", api_key="AIzaSyCWYpI3Q53hDsA8aUQfH2ww9WEkuz_ux5Y")

    test_questions = [
        "سلام، حالت چطوره؟",                          # no need for tool
        "آب‌وهوای تهران الان چطوره؟",                  # calling get_weather
        "حاصل ۲۳۴ ضربدر ۵۶ به‌علاوه ۱۲ چند میشه؟",     # calling caclulate
        "امروز چه برنامه‌ای دارم؟",                     # calling get_calendar_events
    ]

    for q in test_questions:
        print(f"\n{'='*60}")
        print(f"سوال: {q}")
        answer = agent.run(q, verbose=True)
        print(f"\nجواب نهایی: {answer}")