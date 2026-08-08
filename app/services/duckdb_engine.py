import re
import duckdb
import pandas as pd

FORBIDDEN_KEYWORDS = [
    "insert", "update", "delete", "drop", "alter",
    "create", "truncate", "attach", "copy", "pragma", "call",
]


def validate_sql(sql: str) -> None:
    normalized = sql.strip().lower()

    if not normalized.startswith("select"):
        raise ValueError("Only SELECT queries are allowed.")

    if ";" in normalized.rstrip(";"):
        raise ValueError("Multiple statements are not allowed.")

    for keyword in FORBIDDEN_KEYWORDS:
        if re.search(rf"\b{keyword}\b", normalized):
            raise ValueError(f"Query contains forbidden keyword: {keyword}")


def run_query(file_path: str, ext: str, sql: str) -> list[dict]:
    validate_sql(sql)

    if ext == ".csv":
        df = pd.read_csv(file_path)
    else:
        df = pd.read_excel(file_path)

    con = duckdb.connect(database=":memory:")
    con.register("data", df)

    try:
        result_df = con.execute(sql).fetchdf()
    finally:
        con.close()

    return result_df.to_dict(orient="records")