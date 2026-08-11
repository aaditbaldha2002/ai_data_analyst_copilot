from openai import OpenAI
from app.config import settings

client = OpenAI(api_key=settings.openai_api_key)

INTENT_SYSTEM_PROMPT = """Classify the user's question into exactly one category: "forecast", "anomaly", "root_cause", or "historical".

Definitions:
- "forecast": asks to predict/project FUTURE values.
- "anomaly": asks to find unusual/outlier/suspicious individual data points.
- "root_cause": asks WHY a metric changed, increased, decreased, dropped, or spiked — especially referencing \
a specific time period, direction of change, or cause. The presence of "why" together with a change word \
(drop, increase, spike, decline, fell, rose, changed) strongly signals root_cause, even if it also mentions a date.
- "historical": any other question about existing/past data (totals, trends, comparisons, rankings) where \
the user is NOT asking "why" something happened.

Examples:
Q: "Why did revenue drop in April?" -> root_cause
Q: "What caused the increase in signups last quarter?" -> root_cause
Q: "Why is churn up this month?" -> root_cause
Q: "Show monthly revenue trend" -> historical
Q: "What was total revenue in April?" -> historical
Q: "Which product sold the most?" -> historical
Q: "Predict revenue for next quarter" -> forecast
Q: "Show unusual transactions" -> anomaly

Respond with ONLY one word: "forecast", "anomaly", "root_cause", or "historical".
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
    print(f"DEBUG intent classification for '{question}' -> raw answer: '{answer}'")
    if "forecast" in answer:
        return "forecast"
    if "anomaly" in answer:
        return "anomaly"
    if "root_cause" in answer or "root cause" in answer:
        return "root_cause"
    return "historical"