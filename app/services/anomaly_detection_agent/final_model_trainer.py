# ============================================================
# 9. FINAL MODEL TRAINER
# ============================================================

from typing import Any

from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

from app.services.graph_state import GraphState


def final_model_trainer(graph: GraphState) -> GraphState:
    """
    Trains the final anomaly detection models using the selected
    hyperparameters.

    The current implementation keeps all optimized candidates.
    A future model-selection node can choose one final model or
    construct an ensemble.
    """

    X = graph.get("transformed_features")

    if X is None:
        return {
            "final_models": {},
            "warning": "No transformed features available.",
        }

    optimized_params = graph.get(
        "optimized_params",
        {}
    )

    final_models: dict[str, Any] = {}

    for model_name, params in optimized_params.items():

        if model_name == "isolation_forest":

            model = IsolationForest(
                **params
            )

            model.fit(X)

            final_models[model_name] = model

        elif model_name == "lof":

            model = LocalOutlierFactor(
                novelty=True,
                **params,
            )

            model.fit(X)

            final_models[model_name] = model

    return {
        "final_models": final_models,
    }
