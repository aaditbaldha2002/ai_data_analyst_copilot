
# ============================================================
# 11. ANOMALY EXPLAINER
# ============================================================

import pandas as pd

from app.services.graph_state import GraphState


def anomaly_explainer(graph: GraphState) -> GraphState:
    """
    Produces basic explanations for detected anomalies.

    The current implementation identifies the original feature
    values associated with anomalous observations.

    A production version can use:
        - feature contribution analysis
        - SHAP
        - distance-based explanations
        - reconstruction error
        - feature deviation from normal population
    """

    df = graph.get("dataframe")
    anomaly_results = graph.get(
        "anomaly_results",
        []
    )

    if df is None:
        return {
            "anomaly_explanations": []
        }

    result_df = pd.DataFrame(
        anomaly_results
    )

    if "is_anomaly" not in result_df.columns:
        return {
            "anomaly_explanations": []
        }

    feature_columns = graph.get(
        "feature_columns",
        []
    )

    explanations = []

    anomaly_rows = result_df[
        result_df["is_anomaly"]
    ]

    for index, row in anomaly_rows.iterrows():

        explanation = {
            "row_index": int(index),
            "is_anomaly": True,
            "features": {
                feature: row.get(feature)
                for feature in feature_columns
            },
            "model_agreement": row.get(
                "model_agreement"
            ),
        }

        explanations.append(
            explanation
        )

    return {
        "anomaly_explanations": explanations,
    }
