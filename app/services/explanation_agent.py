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

FORECAST_EXPLANATION_SYSTEM_PROMPT = """You are a data analyst. You are given a user's forecasting question, \
a summary of historical data, and a set of FUTURE PREDICTED values from a forecasting model.

Write a short, clear explanation (2-4 sentences) in plain English describing what the forecast predicts.

CRITICAL RULES:
- Only describe the values under "Future predictions" as projections/forecasts.
- Never describe historical values as if they were future predictions.
- You may briefly reference the historical trend for context, but be explicit that it is past data.
- Be specific with numbers and dates from the future predictions.
- Do not mention SQL, tables, or column names literally — speak in natural business terms.
"""


def generate_forecast_explanation(
    question: str,
    historical: list[dict],
    predictions: list[dict],
    model_used: str,
) -> str:
    if not predictions:
        return "No forecast could be generated for this question."

    recent_history = historical[-6:] if len(historical) > 6 else historical

    user_prompt = f"""Question: {question}

Recent historical data (past, for context only):
{json.dumps(recent_history, default=str)}

Future predictions (model used: {model_used}):
{json.dumps(predictions, default=str)}

Explanation:"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": FORECAST_EXPLANATION_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.3,
    )

    return response.choices[0].message.content.strip()

ANOMALY_EXPLANATION_SYSTEM_PROMPT = """You are a data analyst. Given a user's question and a set of rows \
where some are flagged as anomalies (is_anomaly: true) and others are normal, write a short (2-4 sentence) \
plain-English summary of what makes the anomalous rows unusual compared to the rest of the data.

Only describe rows where is_anomaly is true as anomalies. Reference specific values where helpful.
"""

def generate_anomaly_explanation(question: str, rows: list[dict]) -> str:
    anomalies = [r for r in rows if r.get("is_anomaly")]
    if not anomalies:
        return "No unusual data points were found."

    sample = anomalies[:15]
    user_prompt = f"""Question: {question}

Anomalous rows found ({len(anomalies)} total, showing up to 15):
{json.dumps(sample, default=str)}

Explanation:"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": ANOMALY_EXPLANATION_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.3,
    )
    return response.choices[0].message.content.strip()