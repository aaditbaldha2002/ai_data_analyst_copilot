from typing import TypedDict, Optional, Any


class GraphState(TypedDict, total=False):
    question: str
    schema: dict
    file_path: str
    ext: str

    intent: str
    sql: str
    target_metric: str
    result: list[dict[str, Any]]

    chart_config: Optional[dict]
    chart_image_path: Optional[str]
    explanation: Optional[str]

    forecast_model_used: Optional[str]
    forecast_predictions: Optional[list]
    anomaly_model_used: Optional[str]
    anomaly_count: Optional[int]
    root_cause_analysis: Optional[dict]

    error: Optional[str]