"""
Week 1 — Custom LangChain Tools
Covers three patterns:
  1. Simple @tool decorator (calculator, file search)
  2. @tool with a Pydantic args_schema (structured input)
  3. BaseTool subclass (stateful — an in-memory note store)

Run this file directly to sanity-check each tool without an LLM involved.
"""

import os
import glob
from typing import Optional

from langchain_core.tools import tool, BaseTool
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# 1. Simple @tool — calculator
# ---------------------------------------------------------------------------
@tool
def calculator(expression: str) -> str:
    """Evaluate a basic arithmetic expression, e.g. '(3 + 4) * 2'.
    Only supports +, -, *, /, ** and parentheses. Do not pass code or imports."""
    allowed_chars = set("0123456789+-*/(). ")
    if not set(expression) <= allowed_chars:
        return "Error: expression contains disallowed characters."
    try:
        return str(eval(expression, {"__builtins__": {}}, {}))
    except Exception as e:
        return f"Error evaluating expression: {e}"


# ---------------------------------------------------------------------------
# 2. Simple @tool — search local files by name
# ---------------------------------------------------------------------------
@tool
def search_codebase(query: str, directory: str = ".") -> str:
    """Search for filenames containing the given query string within a directory
    (recursive). Returns up to 10 matching file paths."""
    matches = []
    for path in glob.glob(f"{directory}/**/*", recursive=True):
        if query.lower() in os.path.basename(path).lower():
            matches.append(path)
        if len(matches) >= 10:
            break
    return "\n".join(matches) if matches else "No matching files found."


# ---------------------------------------------------------------------------
# 3. @tool with a structured Pydantic input schema
# ---------------------------------------------------------------------------
class WeatherInput(BaseModel):
    city: str = Field(description="City name, e.g. 'Hyderabad'")
    units: str = Field(default="celsius", description="'celsius' or 'fahrenheit'")


@tool(args_schema=WeatherInput)
def get_weather(city: str, units: str = "celsius") -> str:
    """Fetch current weather for a given city. Placeholder implementation —
    swap in a real weather API call (e.g. Open-Meteo) later."""
    symbol = "C" if units == "celsius" else "F"
    return f"(placeholder) 27 degrees {symbol} and clear in {city}"


# ---------------------------------------------------------------------------
# 4. BaseTool subclass — stateful in-memory note store
#    Demonstrates a tool that holds state across calls, which the
#    @tool decorator isn't well suited for.
# ---------------------------------------------------------------------------
class NoteStoreInput(BaseModel):
    action: str = Field(description="'add' or 'list'")
    text: Optional[str] = Field(default=None, description="Note text, required if action='add'")


class NoteStoreTool(BaseTool):
    name: str = "note_store"
    description: str = (
        "Add or list short text notes. Use action='add' with text to save a note, "
        "or action='list' to see all saved notes."
    )
    args_schema: type[BaseModel] = NoteStoreInput
    notes: list = []

    def _run(self, action: str, text: Optional[str] = None) -> str:
        if action == "add":
            if not text:
                return "Error: 'text' is required when action='add'."
            self.notes.append(text)
            return f"Saved note #{len(self.notes)}."
        elif action == "list":
            if not self.notes:
                return "No notes saved yet."
            return "\n".join(f"{i+1}. {n}" for i, n in enumerate(self.notes))
        return f"Unknown action: {action}"

    async def _arun(self, action: str, text: Optional[str] = None) -> str:
        return self._run(action, text)


note_store = NoteStoreTool()


# ---------------------------------------------------------------------------
# 5. Placeholder for your code-comment tool — you'll flesh this out once
#    Week 3's fine-tuned model exists. For now it just returns a stub so
#    you can wire the agent together end-to-end today.
# ---------------------------------------------------------------------------
@tool
def generate_comment(code: str) -> str:
    """Generate a short natural-language comment describing what a code
    snippet does. Currently a placeholder — will call the Week 3
    fine-tuned model once it's trained."""
    return "(placeholder) This function does X — replace with real model output."


# ---------------------------------------------------------------------------
# Collect all tools for easy import into agent_demo.ipynb
# ---------------------------------------------------------------------------
ALL_TOOLS = [calculator, search_codebase, get_weather, note_store, generate_comment]


if __name__ == "__main__":
    # Quick manual sanity checks — run `python tools.py`
    print(calculator.invoke({"expression": "(3 + 4) * 2"}))
    print(search_codebase.invoke({"query": "tools", "directory": "."}))
    print(get_weather.invoke({"city": "Hyderabad", "units": "celsius"}))
    print(note_store.invoke({"action": "add", "text": "check week 2 dataset quality"}))
    print(note_store.invoke({"action": "list"}))
    print(generate_comment.invoke({"code": "def add(a, b): return a + b"}))