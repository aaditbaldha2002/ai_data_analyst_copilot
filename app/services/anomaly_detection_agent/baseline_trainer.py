# ============================================================
# 6. BASELINE TRAINER
# ============================================================

from typing import Any

from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from app.services.graph_state import GraphState


def baseline_trainer(graph: GraphState) -> GraphState:
    """
    Trains baseline versions of all candidate anomaly models.

    The purpose of this node is to establish a baseline before
    expensive hyperparameter optimization is performed.
    """

    X = graph.get("transformed_features")

    if X is None:
        return {
            "trained_models": {},
            "warning": "No transformed features available for training.",
        }

    candidate_models = graph.get(
        "candidate_models",
        []
    )

    model_configs = graph.get(
        "model_configs",
        {}
    )

    trained_models: dict[str, Any] = {}

    for model_name in candidate_models:

        params = model_configs.get(
            model_name,
            {}
        ).copy()

        if model_name == "isolation_forest":

            model = IsolationForest(
                **params
            )

            model.fit(X)

            trained_models[model_name] = model

        elif model_name == "lof":

            # novelty=True makes LOF usable in a reusable
            # train -> validate -> predict pipeline.
            model = LocalOutlierFactor(
                novelty=True,
                **params,
            )

            model.fit(X)

            trained_models[model_name] = model

    return {
        "trained_models": trained_models,
    }
