# ============================================================
# 8. HYPERPARAMETER OPTIMIZER
# ============================================================

from app.services.graph_state import GraphState


def hyperparameter_optimizer(graph: GraphState) -> GraphState:
    """
    Performs hyperparameter optimization for candidate models.

    This is intentionally a lightweight placeholder for the first
    implementation.

    A production implementation should use Optuna or another
    optimization framework and an appropriate anomaly-detection
    validation objective.

    IMPORTANT:
        Do not optimize against F1/accuracy unless reliable
        anomaly labels exist.
    """

    candidate_models = graph.get(
        "candidate_models",
        []
    )

    evaluation_results = graph.get(
        "evaluation_results",
        {}
    )

    optimized_params: dict[str, dict] = {}

    for model_name in candidate_models:

        baseline = evaluation_results.get(
            model_name,
            {}
        )

        # Initial implementation:
        # retain baseline parameters.
        #
        # Replace this section with Optuna once the validation
        # objective has been defined.

        if model_name == "isolation_forest":

            optimized_params[model_name] = {
                "n_estimators": 300,
                "contamination": 0.05,
                "random_state": 42,
            }

        elif model_name == "lof":

            profile = graph.get(
                "data_profile",
                {}
            )

            row_count = profile.get(
                "row_count",
                20,
            )

            optimized_params[model_name] = {
                "n_neighbors": min(
                    20,
                    row_count - 1,
                ),
                "contamination": 0.05,
            }

    return {
        "optimized_params": optimized_params,
    }