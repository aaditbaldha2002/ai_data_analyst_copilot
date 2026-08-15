import json
from openai import OpenAI
from pydantic import BaseModel, Field, ValidationError
from typing import Literal
from app.services.graph_state import GraphState
from app.config import settings

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


def anomaly_problem_analyzer(graph: GraphState) -> GraphState:
    user_query = graph.get("user_query", "")

    if not user_query:
        raise ValueError("Anomaly problem analyzer received an empty user query.")

    # Dataset schema should ideally be populated before this node.
    schema = graph.get("schema", {})

    user_prompt = f"User request: {user_query}\n\nAvailable dataset schema: {schema}"

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,
        response_format={"type": "json_object"},
    )

    raw = response.choices[0].message.content

    try:
        parsed = AnomalyProblem.model_validate(json.loads(raw))
    except (json.JSONDecodeError, ValidationError) as e:
        raise ValueError(f"Anomaly problem analyzer returned invalid structured output: {e}\nRaw response: {raw}")

    return {"anomaly_problem": parsed.model_dump()}