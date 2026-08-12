# ============================================================
# 7. MODEL EVALUATOR
# ============================================================

import numpy as np

from app.services.graph_state import GraphState


def model_evaluator(graph: GraphState) -> GraphState:
    """
    Evaluates baseline anomaly detection models.

    For an unsupervised problem, this implementation reports
    model-independent diagnostics rather than pretending that
    accuracy/F1 exists without ground-truth labels.

    Current metrics:
        - anomaly rate
        - score mean
        - score standard deviation

    Future validation strategies can include:
        - labeled evaluation
        - synthetic anomaly injection
        - stability testing
        - domain validation
        - precision@K
    """

    X = graph.get("transformed_features")
    trained_models = graph.get(
        "trained_models",
        {}
    )

    if X is None:
        return {
            "evaluation_results": {},
            "warning": "No transformed features available for evaluation.",
        }

    evaluation_results: dict[str, dict] = {}

    for model_name, model in trained_models.items():

        predictions = model.predict(X)
        scores = model.decision_function(X)

        anomaly_mask = predictions == -1

        evaluation_results[model_name] = {
            "anomaly_count": int(
                anomaly_mask.sum()
            ),
            "anomaly_rate": float(
                anomaly_mask.mean()
            ),
            "score_mean": float(
                np.mean(scores)
            ),
            "score_std": float(
                np.std(scores)
            ),
        }

    return {
        "evaluation_results": evaluation_results,
    }