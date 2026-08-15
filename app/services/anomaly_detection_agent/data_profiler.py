import os
import json
import numpy as np
import pandas as pd
from openai import OpenAI
from pydantic import BaseModel, Field, ValidationError
from typing import Literal

from app.services.graph_state import GraphState
from app.database import SessionLocal
from app.models.datasets import Dataset
from app.config import settings

client = OpenAI(api_key=settings.openai_api_key)


# ---------------------------------------------------------------------------
# Helpers (dtype/date detection — consistent with duckdb_engine.py conventions)
# ---------------------------------------------------------------------------

def _is_stringlike_dtype(series: pd.Series) -> bool:
    """True for pandas 3.0's new 'str' dtype as well as the legacy 'object' dtype."""
    return series.dtype == "object" or str(series.dtype) == "str"


def _looks_like_date(series: pd.Series) -> bool:
    if pd.api.types.is_datetime64_any_dtype(series):
        return True
    if not _is_stringlike_dtype(series):
        return False
    sample = series.dropna().astype(str).iloc[:5]
    if sample.empty:
        return False
    try:
        pd.to_datetime(sample, errors="raise")
        return True
    except (ValueError, TypeError):
        return False


def _fetch_dataset_record(dataset_id: int) -> Dataset | None:
    if not dataset_id:
        return None
    db = SessionLocal()
    try:
        return db.query(Dataset).filter(Dataset.id == dataset_id).first()
    finally:
        db.close()


def _load_full_dataframe(file_path: str) -> pd.DataFrame:
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".csv":
        df = pd.read_csv(file_path)
    else:
        df = pd.read_excel(file_path)

    for column in df.columns:
        if _looks_like_date(df[column]):
            df[column] = pd.to_datetime(df[column], errors="coerce")

    return df


def _evenly_sampled_records(df: pd.DataFrame, n: int = 10) -> list[dict]:
    """Picks n records spread evenly across the dataset (not just the head),
    so the LLM sees representative examples rather than an arbitrary head/tail slice."""
    if len(df) <= n:
        sample_df = df
    else:
        indices = np.linspace(0, len(df) - 1, num=n, dtype=int)
        indices = sorted(set(indices))
        sample_df = df.iloc[indices]

    # Convert to JSON-safe records (datetime -> string, NaN -> None)
    sample_df = sample_df.copy()
    for col in sample_df.columns:
        if pd.api.types.is_datetime64_any_dtype(sample_df[col]):
            sample_df[col] = sample_df[col].astype(str)
    return sample_df.where(pd.notnull(sample_df), None).to_dict(orient="records")


def _numeric_describe(df: pd.DataFrame, numeric_columns: list[str]) -> dict:
    if not numeric_columns:
        return {}
    desc = df[numeric_columns].describe().round(3)
    return desc.to_dict()


def _categorical_cardinality(df: pd.DataFrame, categorical_columns: list[str], top_n: int = 5) -> dict:
    result = {}
    for col in categorical_columns:
        value_counts = df[col].value_counts().head(top_n)
        result[col] = {
            "distinct_count": int(df[col].nunique(dropna=True)),
            "top_values": {str(k): int(v) for k, v in value_counts.items()},
        }
    return result


# ---------------------------------------------------------------------------
# LLM judgment step
# ---------------------------------------------------------------------------

class DataProfileDecision(BaseModel):
    data_sufficient: bool = Field(
        description="Whether this dataset can actually support the stated anomaly-detection objective."
    )
    unmet_requirements: list[str] = Field(
        default_factory=list,
        description="Items from anomaly_problem.profiling_requirements that this data cannot support."
    )
    final_target_columns: list[str] = Field(
        default_factory=list,
        description="Final chosen numeric/target columns for anomaly detection, grounded only in columns that exist."
    )
    final_grouping_columns: list[str] = Field(
        default_factory=list,
        description="Final chosen categorical columns to group/segment anomaly detection by, if any."
    )
    suggested_analysis_dimension: Literal["univariate", "multivariate"]
    contamination_hint: float = Field(
        description="A reasonable estimated contamination rate (0.01-0.3) based on the described distributions and sample records, not a guess."
    )
    data_quality_concerns: list[str] = Field(default_factory=list)
    reasoning: str


DECISION_SYSTEM_PROMPT = """You are a senior data scientist deciding how an anomaly detection pipeline \
should be configured for a specific dataset and request.

You are given:
1. The anomaly problem (objective, requested target/grouping columns, profiling_requirements) from an earlier analysis step.
2. The dataset's full schema (all columns and inferred types).
3. Descriptive statistics (mean/std/min/max/quartiles) for numeric columns.
4. Cardinality and top values for categorical columns.
5. A small number of representative sample records, evenly sampled across the dataset.

Your job is to make concrete, grounded configuration decisions for the anomaly detection step that follows.

Rules:
1. Never invent column names — only use columns present in the schema.
2. Base every decision on the statistics and samples given, not assumptions.
3. If profiling_requirements cannot be satisfied by this data (e.g. "seasonality" requested but no \
   usable datetime column with enough range exists), list them in unmet_requirements.
4. contamination_hint must be a genuine estimate reasoned from the data's spread/outlier-looking values \
   in the samples and describe() statistics, not a default placeholder.
5. Do not claim anomalies exist — you are configuring the pipeline, not detecting anomalies yourself.
6. Do not perform detection calculations. Only reason and decide.

Respond with ONLY a JSON object in this exact shape:
{
  "data_sufficient": true | false,
  "unmet_requirements": ["<requirement>", ...],
  "final_target_columns": ["<column>", ...],
  "final_grouping_columns": ["<column>", ...],
  "suggested_analysis_dimension": "univariate" | "multivariate",
  "contamination_hint": <float between 0.01 and 0.3>,
  "data_quality_concerns": ["<concern>", ...],
  "reasoning": "<string>"
}
"""


def _decide_profile_configuration(
    anomaly_problem: dict,
    schema: dict,
    numeric_describe: dict,
    categorical_cardinality: dict,
    sample_records: list[dict],
    row_count: int,
) -> dict:
    user_prompt = f"""Anomaly problem:
{json.dumps(anomaly_problem)}

Dataset schema (all columns):
{json.dumps(schema)}

Total row count: {row_count}

Numeric column statistics (describe):
{json.dumps(numeric_describe)}

Categorical column cardinality and top values:
{json.dumps(categorical_cardinality)}

Sample records (evenly sampled across the dataset):
{json.dumps(sample_records, default=str)}

JSON:"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": DECISION_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,
        response_format={"type": "json_object"},
    )

    raw = response.choices[0].message.content
    try:
        parsed = DataProfileDecision.model_validate(json.loads(raw))
    except (json.JSONDecodeError, ValidationError) as e:
        raise ValueError(f"Data profile decision returned invalid output: {e}\nRaw response: {raw}")

    return parsed.model_dump()


# ---------------------------------------------------------------------------
# Main node
# ---------------------------------------------------------------------------

def data_profiler(graph: GraphState) -> GraphState:
    dataset_id = graph.get("dataset_id")
    dataset = _fetch_dataset_record(dataset_id)

    if not dataset:
        return {
            "data_profile": {
                "valid": False,
                "warning": f"Could not find dataset record for dataset_id={dataset_id}.",
            }
        }

    df = _load_full_dataframe(dataset.file_path)

    if df.empty:
        return {"data_profile": {"valid": False, "warning": "Dataset file is empty."}}

    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    datetime_columns = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
    categorical_columns = [
        c for c in df.columns
        if c not in numeric_columns and c not in datetime_columns
        and (_is_stringlike_dtype(df[c]) or df[c].dtype == "bool")
    ]

    missing_values = {
        column: int(df[column].isna().sum())
        for column in df.columns
        if df[column].isna().any()
    }

    possible_id_columns = [
        column for column in df.columns
        if column.lower().endswith("_id") or column.lower() == "id"
    ]

    numeric_stats = _numeric_describe(df, numeric_columns)
    categorical_cardinality = _categorical_cardinality(df, categorical_columns)
    sample_records = _evenly_sampled_records(df, n=10)

    profile = {
        "valid": True,
        "row_count": len(df),
        "column_count": len(df.columns),
        "columns": df.columns.tolist(),
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "datetime_columns": datetime_columns,
        "missing_values": missing_values,
        "possible_id_columns": possible_id_columns,
        "numeric_stats": numeric_stats,
        "categorical_cardinality": categorical_cardinality,
        "sample_records": sample_records,
    }

    anomaly_problem = graph.get("anomaly_problem", {})
    decision = None
    if anomaly_problem:
        decision = _decide_profile_configuration(
            anomaly_problem=anomaly_problem,
            schema=dataset.schema_json,
            numeric_describe=numeric_stats,
            categorical_cardinality=categorical_cardinality,
            sample_records=sample_records,
            row_count=len(df),
        )

    return {
        "dataframe": df,
        "data_profile": profile,
        "profile_decision": decision,
    }