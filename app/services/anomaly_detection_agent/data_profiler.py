import json
import numpy as np
import pandas as pd
from openai import OpenAI
from pydantic import BaseModel, Field, ValidationError
from typing import Literal
from langgraph.runtime import Runtime

from app.services.graph_state import GraphState
from app.services.graph_context import GraphContext
from app.repositories.dataset_repository import DatasetRepository, DatasetNotFoundError
from app.dataset_ingestion import DatasetSchema
from app.config import settings

client = OpenAI(api_key=settings.openai_api_key)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
#
# NOTE: date/dtype detection helpers that used to live here
# (_is_stringlike_dtype, _looks_like_date) are gone. The dataframe now
# comes from DatasetRepository.get_dataframe(), which reads the already-typed
# Parquet file produced at ingestion — dates are already datetime64,
# numerics are already numeric. Re-inferring types here would risk this
# file's logic drifting out of sync with the centralized version in
# dataset_ingestion.py (the exact bug class already found once with the
# pandas 3.0 dtype=="object" issue). Column *kind* (numeric/categorical/
# date/boolean/id/text) is read from the schema's `kind` tags instead of
# being re-derived from dtypes here.

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

    # NOTE: pandas 3.0's new "str" dtype silently defeats
    # `.where(pd.notnull(df), None)` — the None assignment doesn't stick
    # and null values stay as float NaN, which breaks JSON serialization
    # downstream (NaN isn't valid JSON). Casting to object dtype first
    # makes the None assignment actually take.
    sample_df = sample_df.astype(object).where(pd.notnull(sample_df), None)
    return sample_df.to_dict(orient="records")


def _numeric_describe(df: pd.DataFrame, numeric_columns: list[str]) -> dict:
    if not numeric_columns:
        return {}
    desc = df[numeric_columns].describe().round(3)
    # Same NaN -> None fix as _evenly_sampled_records: an entirely-null
    # numeric column produces NaN for every stat, and json.dumps() on raw
    # NaN emits invalid JSON tokens rather than erroring, so it would
    # silently reach the LLM prompt malformed.
    desc = desc.astype(object).where(pd.notnull(desc), None)
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


def _possible_id_columns(df: pd.DataFrame, schema: DatasetSchema) -> list[str]:
    """
    Combines two signals rather than relying on either alone:
      1. schema-derived kind="id" columns — catches high-cardinality string
         columns (e.g. "customer_email") regardless of naming convention.
      2. name-based heuristic — catches numeric sequential ID columns
         (e.g. "row_id"), which the schema's kind detection classifies as
         "numeric" rather than "id" since that check only runs on
         non-numeric columns (see dataset_ingestion.py's _infer_column_kind).
    Neither signal alone covers both cases, so both are kept.
    """
    schema_id_columns = set(schema.column_names("id"))
    name_based = {
        column for column in df.columns
        if column.lower().endswith("_id") or column.lower() == "id"
    }
    return sorted(schema_id_columns | name_based)


# ---------------------------------------------------------------------------
# LLM judgment step (unchanged)
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
    date_column: str | None = Field(
        default=None,
        description=(
            "The single column that represents the time axis for this analysis, if the data is "
            "meaningfully time-ordered and a datetime column exists in the schema. Null if the "
            "dataset isn't a time series or has no usable datetime column. If multiple datetime "
            "columns exist, choose the one that best represents when the observed event/metric "
            "occurred (e.g. an order/transaction date over a signup/account-creation timestamp)."
        )
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
7. date_column must be an actual datetime-typed column from the schema, or null. Only set it if the data \
   is genuinely time-ordered in a way relevant to the anomaly objective — don't pick a date column just \
   because one exists in the schema.

Respond with ONLY a JSON object in this exact shape:
{
  "data_sufficient": true | false,
  "unmet_requirements": ["<requirement>", ...],
  "final_target_columns": ["<column>", ...],
  "final_grouping_columns": ["<column>", ...],
  "date_column": "<column>" | null,
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

    # Same hallucination guard used elsewhere in the pipeline (e.g.
    # anomaly_problem_analyzer's column validation): never trust an
    # LLM-picked column name without checking it's a real datetime-kind
    # column in the schema.
    valid_date_columns = {c["name"] for c in schema.get("columns", []) if c.get("kind") == "date"}
    if parsed.date_column not in valid_date_columns:
        parsed.date_column = None

    return parsed.model_dump()


# ---------------------------------------------------------------------------
# Main node
# ---------------------------------------------------------------------------

def data_profiler(graph: GraphState, runtime: Runtime[GraphContext]) -> dict:
    dataset_id = graph.get("dataset_id")

    if not dataset_id:
        return {
            "data_profile": {
                "valid": False,
                "warning": "No dataset_id present in graph state.",
            }
        }

    try:
        db = runtime.context.db
        repo = DatasetRepository(db)
        try:
            schema = repo.get_schema(dataset_id)      # dataset_columns table — single source of truth for column kinds
            df = repo.get_dataframe(dataset_id)        # full width, per Option A — already-typed Parquet, no re-inference needed
        except DatasetNotFoundError:
            return {
                "data_profile": {
                    "valid": False,
                    "warning": f"Could not find processed data for dataset_id={dataset_id}.",
                }
            }

        if df.empty:
            return {"data_profile": {"valid": False, "warning": "Dataset file is empty."}}

        numeric_columns = schema.column_names("numeric")
        datetime_columns = schema.column_names("date")
        boolean_columns = schema.column_names("boolean")
        categorical_columns = schema.column_names("categorical") + boolean_columns

        missing_values = {
            column: int(df[column].isna().sum())
            for column in df.columns
            if df[column].isna().any()
        }

        possible_id_columns = _possible_id_columns(df, schema)

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
            "boolean_columns": boolean_columns,
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
                schema=schema.to_dict(),
                numeric_describe=numeric_stats,
                categorical_cardinality=categorical_cardinality,
                sample_records=sample_records,
                row_count=len(df),
            )

        # NOTE: the dataframe itself is deliberately NOT returned into
        # graph state. State moves from node to node (and may be
        # checkpointed by LangGraph between steps) — carrying a full
        # dataframe through it adds memory overhead for no benefit, since
        # any downstream node can re-fetch the same data cheaply via
        # DatasetRepository.get_dataframe(dataset_id). Only dataset_id
        # and JSON-safe results travel through state.
        return {
            "data_profile": profile,
            "profile_decision": decision,
        }
    except Exception as e:
        return {"error": str(e)}