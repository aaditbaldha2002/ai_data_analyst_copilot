# ============================================================
# 10. ANOMALY SCORER
# ============================================================

import numpy as np

from app.services.graph_state import GraphState


def anomaly_scorer(graph: GraphState) -> GraphState:
    """
    Generates anomaly predictions and scores from the final models.

    Output:
        - Individual model predictions
        - Individual model scores
        - Ensemble anomaly flag
        - Model agreement score
    """

    df = graph.get("dataframe")
    X = graph.get("transformed_features")
    final_models = graph.get(
        "final_models",
        {}
    )

    if df is None or X is None or not final_models:
        return {
            "anomaly_results": [],
            "anomaly_count": 0,
            "warning": "No trained anomaly models available.",
        }

    predictions: dict[str, np.ndarray] = {}
    scores: dict[str, np.ndarray] = {}

    for model_name, model in final_models.items():

        predictions[model_name] = model.predict(X)

        scores[model_name] = (
            model.decision_function(X)
        )

    prediction_matrix = np.vstack([
        prediction == -1
        for prediction in predictions.values()
    ])

    # Initial ensemble strategy:
    # majority vote.
    anomaly_flags = (
        prediction_matrix.sum(axis=0)
        >= prediction_matrix.shape[0] / 2
    )

    model_agreement = (
        prediction_matrix.mean(axis=0)
    )

    result_df = df.copy()

    result_df["is_anomaly"] = anomaly_flags

    result_df["model_agreement"] = model_agreement

    for model_name, model_predictions in predictions.items():

        result_df[
            f"{model_name}_anomaly"
        ] = model_predictions == -1

    for model_name, model_scores in scores.items():

        result_df[
            f"{model_name}_score"
        ] = model_scores

    anomaly_count = int(
        result_df["is_anomaly"].sum()
    )

    return {
        "anomaly_results": result_df.to_dict(
            orient="records"
        ),
        "anomaly_count": anomaly_count,
        "model_predictions": predictions,
        "model_scores": scores,
    }
