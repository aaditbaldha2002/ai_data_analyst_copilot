# ============================================================
# 12. RESPONSE GENERATOR
# ============================================================

from app.services.graph_state import GraphState


def response_generator(graph: GraphState) -> GraphState:
    """
    Generates the final structured response for the parent graph.

    This node should eventually be responsible for converting
    technical anomaly detection results into a concise response
    appropriate for the user.

    An LLM can be introduced here for natural-language generation.
    """

    anomaly_count = graph.get(
        "anomaly_count",
        0,
    )

    total_rows = graph.get(
        "data_profile",
        {}
    ).get(
        "row_count",
        0,
    )

    final_models = graph.get(
        "final_models",
        {}
    )

    response = {
        "status": "success",
        "task": "anomaly_detection",
        "total_rows": total_rows,
        "anomaly_count": anomaly_count,
        "anomaly_rate": (
            anomaly_count / total_rows
            if total_rows > 0
            else 0.0
        ),
        "models_used": list(
            final_models.keys()
        ),
        "anomalies": graph.get(
            "anomaly_results",
            []
        ),
        "explanations": graph.get(
            "anomaly_explanations",
            []
        ),
    }

    return {
        "response": response,
    }