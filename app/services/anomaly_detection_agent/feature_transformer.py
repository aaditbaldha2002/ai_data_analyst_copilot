# ============================================================
# 4. FEATURE TRANSFORMER
# ============================================================

import pandas as pd
from sklearn.preprocessing import StandardScaler

from app.services.graph_state import GraphState


def feature_transformer(graph: GraphState) -> GraphState:
    """
    Transforms extracted features into model-ready numerical data.

    Responsibilities:
        - Handle missing values.
        - Scale numerical features.
        - Preserve preprocessing metadata.

    The preprocessing objects should eventually be persisted with
    the trained model for consistent inference.
    """

    X = graph.get("features")

    if X is None or X.empty:
        return {
            "transformed_features": None,
            "warning": "No features available for transformation.",
        }

    X = X.copy()

    # Median imputation.
    for column in X.columns:
        median = X[column].median()

        if pd.isna(median):
            median = 0.0

        X[column] = X[column].fillna(median)

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    return {
        "transformed_features": X_scaled,
        "feature_scaler": scaler,
    }
