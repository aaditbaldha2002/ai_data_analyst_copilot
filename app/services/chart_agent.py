import json
from openai import OpenAI
from app.config import settings

client = OpenAI(api_key=settings.openai_api_key)

CHART_SYSTEM_PROMPT = """You decide how to visualize a SQL query result as a chart.

You are given the user's question and the query result (a list of rows, each a dict of column -> value).

Return ONLY a JSON object with this exact shape, no markdown, no commentary:
{
  "chart_type": "bar" | "line" | "pie" | "none",
  "x_key": "<column name to use for x-axis / labels>",
  "y_key": "<column name to use for y-axis / values>",
  "title": "<short chart title>"
}

Rules:
- Use "line" for trends over time (dates, months, years).
- Use "bar" for comparisons across categories.
- Use "pie" only for simple part-to-whole breakdowns with few categories (<= 6).
- Use "none" if the result has only one row/value (e.g. a single total) or isn't chartable.
- x_key and y_key MUST be exact keys present in the result rows.
"""


def generate_chart_config(question: str, result: list[dict]) -> dict:
    if not result:
        return {"chart_type": "none", "x_key": None, "y_key": None, "title": ""}

    sample = result[:20]  # cap payload size sent to the LLM

    user_prompt = f"""Question: {question}

Result rows (sample):
{json.dumps(sample, default=str)}

JSON:"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": CHART_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,
        response_format={"type": "json_object"},
    )

    raw = response.choices[0].message.content
    try:
        config = json.loads(raw)
    except json.JSONDecodeError:
        config = {"chart_type": "none", "x_key": None, "y_key": None, "title": ""}

    return config