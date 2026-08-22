import json
from typing import Optional

from openai import OpenAI
from pydantic import BaseModel, Field, ValidationError
from typing import Literal
from sqlalchemy.orm import Session
from langgraph.runtime import Runtime

from app.services.graph_state import GraphState
from app.services.graph_context import GraphContext
from app.services.llm_tool_utils import call_llm_tool, LLMToolCallError
from app.config import settings
from app.models.datasets import Dataset
from app.repositories.dataset_repository import DatasetRepository, DatasetNotFoundError

client = OpenAI(api_key=settings.openai_api_key)


class AnomalyProblem(BaseModel):
    objective: str = Field(
        description="What the user wants to detect or investigate."
    )
    anomaly_type: Literal[
        "point", "contextual", "collective", "temporal", "multivariate", "unknown",
    ]
    analysis_dimension: Literal["univariate", "multivariate", "unknown"]
    temporal: bool
    target_columns: list[str] = Field(default_factory=list)
    grouping_columns: list[str] = Field(default_factory=list)
    profiling_requirements: list[
        Literal[
            "schema", "data_types", "missing_values", "cardinality",
            "numeric_distribution", "categorical_distribution", "outliers",
            "correlations", "time_order", "trend", "seasonality",
            "duplicate_records", "group_distribution",
        ]
    ] = Field(default_factory=list)
    reasoning: str


SYSTEM_PROMPT = """
You are an expert data scientist specializing in anomaly detection.

Your task is to analyze the user's anomaly-detection request.

You are NOT performing anomaly detection.

You are determining what the downstream anomaly detection pipeline
needs to investigate.

Determine:
- What the user wants to detect.
- The likely anomaly type.
- Whether the analysis is univariate or multivariate.
- Whether time is relevant.
- Which dataset columns are relevant.
- Whether the analysis should be performed within groups.
- What information the data profiler needs to calculate.

Important rules:
1. Never invent column names.
2. Only use columns supplied in the dataset schema.
3. If the user does not specify a target column, infer one only when
   the schema makes the inference reasonably obvious.
4. If the request is ambiguous, use "unknown".
5. Do not select a machine-learning algorithm.
6. Do not claim that anomalies exist.
7. Do not perform calculations.
8. Return only the requested structured output.

The downstream data profiler will inspect the actual data.

You must respond with ONLY a JSON object (no markdown, no commentary) in this exact shape:
{
  "objective": "<string>",
  "anomaly_type": "point" | "contextual" | "collective" | "temporal" | "multivariate" | "unknown",
  "analysis_dimension": "univariate" | "multivariate" | "unknown",
  "temporal": true | false,
  "target_columns": ["<column name>", ...],
  "grouping_columns": ["<column name>", ...],
  "profiling_requirements": ["schema" | "data_types" | "missing_values" | "cardinality" | "numeric_distribution" | "categorical_distribution" | "outliers" | "correlations" | "time_order" | "trend" | "seasonality" | "duplicate_records" | "group_distribution", ...],
  "reasoning": "<string>"
}
"""


# ---------------------------------------------------------------------------
# Dataset resolution — cheap heuristic first, LLM fallback only if ambiguous
# ---------------------------------------------------------------------------

class _DatasetMatch(BaseModel):
    dataset_id: Optional[int] = None
    confidence: Literal["high", "medium", "low"] = "low"


def _list_user_datasets(db: Session, owner_id: int) -> list[dict]:
    rows = (
        db.query(Dataset)
        .filter(Dataset.owner_id == owner_id, Dataset.parquet_path.isnot(None))
        .order_by(Dataset.created_at.desc())
        .all()
    )
    return [{"id": r.id, "filename": r.filename, "row_count": r.row_count} for r in rows]


def _heuristic_dataset_match(question: str, datasets: list[dict]) -> Optional[dict]:
    q_lower = question.lower()
    for d in datasets:
        name_no_ext = d["filename"].rsplit(".", 1)[0].lower()
        if name_no_ext and name_no_ext in q_lower:
            return d
    return None


def _llm_dataset_match(question: str, datasets: list[dict]) -> Optional[int]:
    system_prompt = (
        "You are matching a user's data question to one of their uploaded datasets. "
        "Call the submit_dataset_match tool with your answer. "
        "Use null for dataset_id if no dataset is a plausible match."
    )
    user_prompt = f"User question: {question}\n\nAvailable datasets: {json.dumps(datasets)}"

    try:
        parsed = call_llm_tool(
            client,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_model=_DatasetMatch,
            tool_name="submit_dataset_match",
            tool_description="Submit which dataset (by id) the user's question most likely refers to.",
        )
    except LLMToolCallError:
        return None

    if parsed.confidence == "low":
        return None
    return parsed.dataset_id


def _resolve_dataset(db: Session, owner_id: int, question: str) -> Optional[dict]:
    """
    Returns {"id": ..., "filename": ..., "row_count": ...} for the dataset
    this question most likely refers to, or None if it can't be resolved
    (no datasets uploaded, or genuinely ambiguous with low LLM confidence).
    """
    datasets = _list_user_datasets(db, owner_id)
    if not datasets:
        return None
    if len(datasets) == 1:
        return datasets[0]

    match = _heuristic_dataset_match(question, datasets)
    if match:
        return match

    matched_id = _llm_dataset_match(question, datasets)
    if matched_id is None:
        return None

    for d in datasets:
        if d["id"] == matched_id:
            return d
    return None


def anomaly_problem_analyzer(state: GraphState, runtime: Runtime[GraphContext]) -> dict:
    """
    Entry node of the anomaly subgraph.

    1. Resolves which of the user's datasets the question refers to
       (cheap filename-substring match first, LLM fallback if ambiguous).
    2. Fetches that dataset's real schema via DatasetRepository — never
       from a stale/legacy state field.
    3. Asks the LLM to analyze the anomaly-detection objective (target
       columns, grouping, temporal relevance, profiling requirements)
       grounded in that real schema.

    Uses the db session from GraphContext rather than opening its own —
    the context exists specifically so nodes don't each manage their own
    session lifecycle.

    Returns {"error": ...} on failure rather than raising, consistent
    with every other node in this pipeline — an uncaught exception here
    would crash the whole graph run instead of letting finalize_node
    handle it gracefully.
    """
    try:
        question = state.get("question", "")
        owner_id = state.get("owner_id")

        if not question:
            return {"error": "Anomaly problem analyzer received an empty question."}
        if not owner_id:
            return {"error": "No owner_id present in graph state."}

        db = runtime.context.db

        dataset = _resolve_dataset(db, owner_id, question)
        if dataset is None:
            return {"error": "Could not determine which dataset this question refers to."}

        repo = DatasetRepository(db)
        try:
            schema = repo.get_schema(dataset["id"])
        except DatasetNotFoundError:
            return {"error": f"Dataset '{dataset['filename']}' has no processed data available."}

        schema_dict = schema.to_dict()
        user_prompt = f"User request: {question}\n\nAvailable dataset schema: {json.dumps(schema_dict)}"

        try:
            parsed = call_llm_tool(
                client,
                system_prompt=SYSTEM_PROMPT,
                user_prompt=user_prompt,
                response_model=AnomalyProblem,
                tool_name="submit_anomaly_problem",
                tool_description="Submit the structured analysis of the user's anomaly-detection request.",
            )
        except LLMToolCallError as e:
            return {"error": f"Anomaly problem analyzer returned invalid structured output: {e}"}

        # Same hallucination guard used throughout the rest of this
        # pipeline — the prompt already instructs the LLM not to invent
        # column names, but a prompt instruction alone isn't a guarantee.
        valid_columns = {c["name"] for c in schema_dict["columns"]}
        problem = parsed.model_dump()
        problem["target_columns"] = [c for c in problem["target_columns"] if c in valid_columns]
        problem["grouping_columns"] = [c for c in problem["grouping_columns"] if c in valid_columns]

        return {
            "dataset_id": dataset["id"],
            "anomaly_problem": problem,
        }
    except Exception as e:
        return {"error": str(e)}