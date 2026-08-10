import json
from openai import OpenAI
from app.config import settings

client = OpenAI(api_key=settings.openai_api_key)

EXPLANATION_SYSTEM_PROMPT = """You are a data analyst. Given a user's question and the SQL query result, \
write a short, clear explanation (2-4 sentences) of the finding in plain English.

Do not mention SQL, tables, or column names literally — speak in natural business terms.
Be specific with numbers where relevant. Do not speculate beyond what the data shows.

CRITICAL: Before writing, carefully compare the actual numeric values in the result data. \
Double-check any ranking, comparison, or "highest/lowest/leading" claim against the real numbers — \
do not assume the row order in the data reflects rank unless it is explicitly sorted. \
State comparisons only when they are numerically correct.
"""


def generate_explanation(question: str, result: list[dict]) -> str:
    if not result:
        return "No results were found for this question."

    sample = _sort_for_readability(result)[:30]

    user_prompt = f"""Question: {question}

Result data (sorted by value, descending, where applicable):
{json.dumps(sample, default=str)}

Explanation:"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": EXPLANATION_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.3,
    )

    return response.choices[0].message.content.strip()


def _sort_for_readability(result: list[dict]) -> list[dict]:
    numeric_keys = [
        k for k, v in result[0].items()
        if isinstance(v, (int, float)) and not isinstance(v, bool)
    ]
    if not numeric_keys:
        return result
    sort_key = numeric_keys[0]
    try:
        return sorted(result, key=lambda row: row.get(sort_key, 0), reverse=True)
    except TypeError:
        return result