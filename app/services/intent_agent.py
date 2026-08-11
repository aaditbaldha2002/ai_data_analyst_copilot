from openai import OpenAI
from app.config import settings

client = OpenAI(api_key=settings.openai_api_key)

INTENT_SYSTEM_PROMPT = """Classify the user's question as one of: "forecast", "anomaly", or "historical".

"forecast" = predicting/projecting future values (e.g. "predict next month's sales").
"anomaly" = finding unusual, outlier, or suspicious data points (e.g. "show unusual transactions", "find outliers in revenue", "which entries look suspicious").
"historical" = anything else about existing/past data (aggregations, comparisons, trends already in the data).

Respond with ONLY one word: "forecast", "anomaly", or "historical".
"""

def classify_intent(question: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": INTENT_SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
        temperature=0,
    )
    answer = response.choices[0].message.content.strip().lower()
    if "forecast" in answer:
        return "forecast"
    if "anomaly" in answer:
        return "anomaly"
    return "historical"