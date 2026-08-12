# ============================================================
# 5. MODEL SELECTOR
# ============================================================

from app.services.graph_state import GraphState


def model_selector(graph: GraphState) -> GraphState:
    """
    Selects candidate anomaly detection algorithms.

    The initial implementation supports:
        - Isolation Forest
        - Local Outlier Factor

    Later this can incorporate:
        - One-Class SVM
        - Elliptic Envelope
        - DBSCAN
        - Autoencoders
        - PyOD models
        - Domain-specific detectors
    """

    profile = graph.get("data_profile", {})
    problem = graph.get("anomaly_problem", {})

    row_count = profile.get("row_count", 0)
    validation_strategy = problem.get(
        "validation_strategy",
        "unsupervised",
    )

    if row_count < 5:
        return {
            "candidate_models": [],
            "warning": (
                "Not enough observations for reliable "
                "anomaly detection."
            ),
        }

    candidate_models = [
        "isolation_forest"
    ]

    # LOF requires a reasonable number of observations.
    if row_count >= 20:
        candidate_models.append("lof")

    model_configs = {
        "isolation_forest": {
            "contamination": 0.05,
            "n_estimators": 200,
            "random_state": 42,
        },
        "lof": {
            "n_neighbors": min(20, row_count - 1),
            "contamination": 0.05,
        },
    }

    return {
        "candidate_models": candidate_models,
        "model_configs": model_configs,
        "validation_strategy": validation_strategy,
    }
