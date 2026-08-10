from openai import OpenAI
from app.config import settings

client = OpenAI(api_key=settings.openai_api_key)

INTENT_SYSTEM_PROMPT = """Classify the user's question as either "forecast" or "historical".

"forecast" = the user is asking to predict, project, or estimate future values \
(e.g. "predict next month's sales", "what will revenue be in Q3", "forecast demand for next 6 months").

"historical" = the user is asking about existing/past data \
(e.g. "show monthly revenue", "which product sold the most", "total sales last year").

Respond with ONLY one word: "forecast" or "historical".
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
    return "forecast" if "forecast" in answer else "historical"