from app.services.graph_state import GraphState

def anomaly_problem_analyzer(graph: GraphState) -> GraphState:
    """
    Analyzes the user's anomaly-detection request and determines
    the characteristics of the anomaly detection task.

    Responsibilities:
        - Identify the anomaly detection objective.
        - Determine whether the task is likely point, contextual,
          collective, temporal, or multivariate anomaly detection.
        - Determine whether labels are available.
        - Establish initial assumptions for downstream ML nodes.

    This node should eventually be enhanced with an LLM or a
    dedicated problem-classification component.
    """

    user_query = graph.get("user_query", "")
    data = graph.get("data", graph.get("result", []))

    # Basic task analysis for the initial implementation.
    task_type = "tabular_unsupervised"

    if isinstance(data, list):
        row_count = len(data)
    else:
        row_count = 0

    labels_available = False

    if isinstance(data, list) and data:
        columns = set(data[0].keys())

        possible_label_columns = {
            "label",
            "is_anomaly",
            "anomaly",
            "anomaly_label",
            "fraud",
            "is_fraud",
        }

        labels_available = bool(
            columns.intersection(possible_label_columns)
        )

    if labels_available:
        validation_strategy = "supervised"
    else:
        validation_strategy = "unsupervised"

    return {
        "anomaly_problem": {
            "task_type": task_type,
            "labels_available": labels_available,
            "validation_strategy": validation_strategy,
            "row_count": row_count,
            "user_query": user_query,
        }
    }
