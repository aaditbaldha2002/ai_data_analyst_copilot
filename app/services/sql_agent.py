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

ANOMALY_SQL_SYSTEM_PROMPT = """You are a SQL generation assistant. The user wants to find unusual/anomalous \
rows in their data. Generate a SELECT query that returns the RAW ROWS needed for anomaly detection — \
do NOT aggregate (no GROUP BY, no SUM/AVG/COUNT) unless the user explicitly asks about a specific \
aggregated metric being unusual.

Rules:
- Only generate SELECT statements.
- Only reference the table named "data".
- Prefer returning all relevant numeric and identifying columns so outliers can be detected across dimensions.
- If the question mentions a specific column or subset, filter appropriately, otherwise select broadly.
- Limit to at most 1000 rows (add LIMIT 1000) to keep this bounded.
- Return ONLY the raw SQL query. No explanations, no markdown fences.
"""

def generate_anomaly_sql(question: str, schema: dict) -> str:
    schema_description = "\n".join(f"- {col}: {dtype}" for col, dtype in schema.items())
    user_prompt = f"""Table columns:
{schema_description}

Question: {question}

SQL query:"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": ANOMALY_SQL_SYSTEM_PROMPT},
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

import json  # add this import at the top if not already present

ROOT_CAUSE_SQL_SYSTEM_PROMPT = """You are a SQL generation assistant. The user wants to understand WHY \
a metric changed (e.g. dropped or increased) over some time period.

Generate a SELECT query that returns ROW-LEVEL data (not aggregated) covering a reasonable window \
around the period mentioned — include enough prior context to compare against (e.g. if the question \
mentions April, include at least February through April so a before/after comparison is possible).

Rules for the SQL:
- Only generate SELECT statements.
- Only reference the table named "data".
- Do NOT aggregate with GROUP BY/SUM/AVG — return raw rows with all relevant columns.
- Add LIMIT 2000 to keep this bounded.

You must respond with ONLY a JSON object (no markdown, no commentary) in this exact shape:
{
  "sql": "<the raw SELECT query>",
  "target_metric": "<the exact column name from the schema that represents the metric the user is asking about>"
}
"""


def generate_root_cause_sql(question: str, schema: dict) -> dict:
    schema_description = "\n".join(f"- {col}: {dtype}" for col, dtype in schema.items())
    user_prompt = f"""Table columns:
{schema_description}

Question: {question}

JSON:"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": ROOT_CAUSE_SQL_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,
        response_format={"type": "json_object"},
    )
    return json.loads(response.choices[0].message.content)