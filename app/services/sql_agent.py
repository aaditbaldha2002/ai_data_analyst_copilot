from openai import OpenAI
from app.config import settings

client = OpenAI(api_key=settings.openai_api_key)

SYSTEM_PROMPT = """You are a SQL generation assistant. You are given a table named "data" \
with the following columns and types, and a user's natural language question.

Generate a single, valid DuckDB SQL query that answers the question.

Rules:
- Only generate SELECT statements. Never generate INSERT, UPDATE, DELETE, DROP, ALTER, or any other statement.
- Only reference the table named "data".
- This is DuckDB SQL. For date formatting, use strftime(column, format) — the DATE/TIMESTAMP column comes FIRST, the format string comes SECOND. Example: strftime(date_column, '%Y-%m').
- Return ONLY the raw SQL query. No explanations, no markdown code fences, no commentary.
"""

def generate_sql(question: str, schema: dict) -> str:
    schema_description = "\n".join(f"- {col}: {dtype}" for col, dtype in schema.items())

    user_prompt = f"""Table columns:
{schema_description}

Question: {question}

SQL query:"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,
    )

    sql = response.choices[0].message.content.strip()

    # Strip markdown fences if the model adds them despite instructions
    if sql.startswith("```"):
        sql = sql.strip("`")
        if sql.lower().startswith("sql"):
            sql = sql[3:].strip()

    return sql

FORECAST_SQL_SYSTEM_PROMPT = """You are a SQL generation assistant. You are given a table named "data" \
with the following columns and types, and a user's question that asks for a FUTURE prediction.

Your job is NOT to predict anything. Instead, generate a SELECT query that returns the relevant \
HISTORICAL time series needed to make that prediction — grouped by an appropriate time period \
(e.g. by month), with a date/period column and a numeric value column.

Rules:
- Only generate SELECT statements.
- Only reference the table named "data".
- This is DuckDB SQL. For date formatting, use strftime(column, format) — the DATE/TIMESTAMP column comes FIRST, the format string comes SECOND.
- Always include ORDER BY the date/period column ascending.
- Return ONLY the raw SQL query. No explanations, no markdown code fences, no commentary.
"""


def generate_forecast_sql(question: str, schema: dict) -> str:
    schema_description = "\n".join(f"- {col}: {dtype}" for col, dtype in schema.items())

    user_prompt = f"""Table columns:
{schema_description}

Question: {question}

SQL query:"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": FORECAST_SQL_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,
    )

    sql = response.choices[0].message.content.strip()
    if sql.startswith("```"):
        sql = sql.strip("`")
        if sql.lower().startswith("sql"):
            sql = sql[3:].strip()

    return sql