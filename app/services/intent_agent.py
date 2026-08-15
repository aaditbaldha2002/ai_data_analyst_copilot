from openai import OpenAI
from sqlalchemy.orm import Session, joinedload

from app.config import settings
from app.models.datasets import Dataset


client = OpenAI(api_key=settings.openai_api_key)


INTENT_SYSTEM_PROMPT = """
Classify the user's question into exactly one category:
"forecast", "anomaly", "root_cause", or "historical".

Also identify the dataset to which this query is targeted.

The available datasets for this user will be provided under the
<list_of_datasets> tag.

Dataset selection rules:

1. If the name of the dataset or filename is explicitly mentioned
   in the user's question, prefer that dataset.
2. Otherwise, identify the most appropriate dataset based on:
   - filename
   - column names
   - column data types
   - column kinds
   - row count
3. You MUST select the dataset only from the datasets provided under
   <list_of_datasets>.
4. NEVER invent a dataset ID.
5. If the dataset cannot be identified with reasonable confidence,
   return null for dataset_id.

Definitions:
- "forecast":
  asks to predict/project FUTURE values.
- "anomaly":
  asks to find unusual/outlier/suspicious individual data points.
- "root_cause":
  asks WHY a metric changed, increased, decreased, dropped, or spiked —
  especially referencing a specific time period, direction of change,
  or cause.
  The presence of "why" together with a change word
  (drop, increase, spike, decline, fell, rose, changed)
  strongly signals root_cause, even if it also mentions a date.
- "historical":
  any other question about existing/past data
  (totals, trends, comparisons, rankings) where the user is NOT
  asking "why" something happened.
Examples:
Q: "Why did revenue drop in April?"
-> root_cause
Q: "What caused the increase in signups last quarter?"
-> root_cause
Q: "Why is churn up this month?"
-> root_cause
Q: "Show monthly revenue trend"
-> historical
Q: "What was total revenue in April?"
-> historical
Q: "Which product sold the most?"
-> historical
Q: "Predict revenue for next quarter"
-> forecast
Q: "Show unusual transactions"
-> anomaly

Respond in this exact format: 
{"intent": "forecast","dataset_id": 123}
"""


def classify_intent(
    question: str,
    owner_id: int,
    db: Session,
) -> dict:

    datasets = (
        db.query(Dataset)
        .options(joinedload(Dataset.columns))
        .filter(Dataset.owner_id == owner_id)
        .all()
    )

    dataset_list = []

    for dataset in datasets:
        columns = []
        for column in dataset.columns:
            columns.append({
                "name": column.name,
                "dtype": column.dtype,
                "kind": column.kind,
            })
        dataset_list.append({
            "dataset_id": dataset.id,
            "filename": dataset.filename,
            "row_count": dataset.row_count,
            "columns": columns,
        })

    user_prompt = f"""
        <list_of_datasets>
        {dataset_list}
        </list_of_datasets>

        <user_question>
        {question}
        </user_question>
        """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": INTENT_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
        temperature=0,
    )

    raw_answer = response.choices[0].message.content.strip()

    import json

    try:
        result = json.loads(raw_answer)
    except json.JSONDecodeError:
        raise ValueError(
            f"LLM returned invalid classification JSON: "
            f"{raw_answer}"
        )

    intent = result.get("intent")
    dataset_id = result.get("dataset_id")

    valid_intents = {
        "forecast",
        "anomaly",
        "root_cause",
        "historical",
    }

    if intent not in valid_intents:
        raise ValueError(
            f"Invalid intent returned by LLM: {intent}"
        )

    if dataset_id is not None:
        dataset = (
            db.query(Dataset)
            .filter(
                Dataset.id == dataset_id,
                Dataset.owner_id == owner_id,
            )
            .first()
        )

        if dataset is None:
            raise PermissionError(
                "The selected dataset does not belong "
                "to the current user."
            )

    return {
        "intent": intent,
        "dataset_id": dataset_id,
    }