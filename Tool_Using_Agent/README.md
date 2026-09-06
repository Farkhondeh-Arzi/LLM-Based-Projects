# Tool-Using Agent (LLM Function Calling)

An implementation of an **Agent** that, instead of merely generating text, can detect when it needs an external tool (function), call it with the correct arguments, receive the result, and build the final response based on that result.

This project implements the **Function Calling / Tool Use** concept with two different backends: **Local Llama (free, via Ollama)** and **Gemini API (Google)** — to demonstrate that the core agent logic is independent of the provider, and only the API format details differ.

## Why This Project?

A regular LLM only responds based on the knowledge it was trained on and has no access to real-time data or precise calculations. Function calling solves this limitation: the model detects when it should use a real tool (weather, calculator, calendar, or any other API), and our code executes that tool and returns the result to the model so it can build an accurate final response.

## Project Structure

```
tool_agent/
├── requirements.txt
├── README.md
└── src/
    ├── tools.py              # Tool definitions and their schemas
    ├── agent_loop.py         # Agent loop with local Llama (Ollama)
    └── agent_loop_gemini.py  # Agent loop with Gemini API (Google)
```

## `tools.py` — Tool Definitions

This file is provider-independent and consists of three parts:

### 1. Actual Functions (Simulated)

Three tools are implemented:

| Tool | What It Does | Technical Note |
|---|---|---|
| `get_weather(city)` | Returns simulated weather for a city | Generates random but plausible data, no real API needed |
| `calculate(expression)` | Performs accurate mathematical calculation | Implemented with the `ast` module, **not** raw `eval()` — because directly executing model input with `eval()` is a security risk |
| `get_calendar_events(date)` | Returns events for a given day (`today`/`tomorrow`) | Reads from a simple fake database |

### 2. `AVAILABLE_TOOLS`

A dictionary that maps tool names to their actual Python functions. This dictionary is used in the agent loop so that when the model announces a tool name, the correct function is executed — and **only** tools in this dictionary are allowed to run (a security defense layer).

### 3. `TOOL_SCHEMAS` and `get_gemini_tool_schemas()`

`TOOL_SCHEMAS` stores the description of each tool in the standard OpenAI-style format (which Ollama also supports) — this is what gets "introduced" to the model so it understands what each tool does and what parameters it expects.

Since Gemini uses a flatter format (without the nested `"function"` structure), the `get_gemini_tool_schemas()` function automatically performs this conversion. This means tools are defined **once** and work for both providers — with no code duplication.

## `agent_loop.py` — Local Llama Version (Ollama)

The `ToolAgent` class implements the core agent logic using the `ollama` library:

- **`execute_tool_call()`** — Security layer that only executes tools present in `AVAILABLE_TOOLS` and gracefully handles errors.
- **`ToolAgent.run()`** — The main loop:
  1. The user's question and tool list are sent to the model.
  2. If the model requests a tool (`message["tool_calls"]`), the actual function is executed and the result is added to the `messages` history with `role="tool"`, then we loop back to the model.
  3. If the model returns a final text response, the loop exits with `return`.
- **`max_iterations`** — A safeguard against infinite loops, for cases where the model keeps requesting tools and never reaches a final answer.

No cost or API key required; runs completely offline on your own system (prerequisite: install [Ollama](https://ollama.com) and download the model with `ollama pull llama3.1:8b`).

## `agent_loop_gemini.py` — Gemini API Version

The `GeminiToolAgent` class implements the same logic using Google's new **Interactions API** (2026). Key differences from the Ollama version:

| Concept | Ollama | Gemini |
|---|---|---|
| Model response unit | A single message | List of `steps` (`function_call`, `text`, ...) |
| History management | List of `messages` | List of `history` (stateless mode, with `store=False`) |
| Returning tool result | Message with `role="tool"` | A `function_result` step with matching `call_id` |
| Request/result matching | Based solely on order | Requires explicit `call_id` |

Important learning point: **The core agent logic (decide → execute → return result → repeat) is completely provider-independent.** Only the API format details differ — this is what you need to know when working with different LLM providers in the real world.

Requires a free API key (with rate limits) from [aistudio.google.com/apikey](https://aistudio.google.com/apikey).

## How to Run

```bash
pip install -r requirements.txt

# For local Llama version:
# 1. Install Ollama (ollama.com)
# 2. ollama pull llama3.1:8b
cd src
python3 agent_loop.py

# Or for Gemini version:
export GEMINI_API_KEY="your_key"
python3 agent_loop_gemini.py
```

## Sample Real Output (Tested with Gemini)

```
Question: What's the weather like in Tehran right now?
Model requested tool: get_weather({'city': 'Tehran'})
Tool result: {'city': 'Tehran', 'temperature_celsius': 36, 'condition': 'Dusty'}
Final answer: Currently, the weather in Tehran is dusty with a temperature of 36°C.

Question: What is 234 times 56 plus 12?
Model requested tool: calculate({'expression': '234 * 56 + 12'})
Tool result: {'expression': '234 * 56 + 12', 'result': 13116}
Final answer: 234 times 56 plus 12 equals 13,116.
```

You can see that the second question (casual greeting) didn't call any tools and was answered directly, while the weather and math questions were correctly identified as requiring tools — this is exactly the behavior expected from a good agent.

## Design Highlights Worth Mentioning in Interviews

- **Separation of decision-making from execution:** The model never directly executes code; it only says "call this function with these arguments," and our code decides whether this action is allowed or not.
- **Security in `calculate`:** Using `ast` instead of raw `eval()` to prevent arbitrary code execution.
- **`max_iterations` safeguard:** Prevents infinite loops and uncontrolled costs in case of unexpected model behavior.
- **Provider-agnostic architecture:** Tools are defined once and are reusable across two different backends (Ollama, Gemini) with just a small conversion function.

## Known Limitations (For Future Development)

- Tools are simulated; in a real version, they should connect to actual APIs (OpenWeatherMap, Google Calendar, etc.).
- No authentication/authorization mechanism exists for tools — in a production environment, you need to specify which user has permission to execute which tool.
- Network errors (timeout, API unavailability) are not handled.