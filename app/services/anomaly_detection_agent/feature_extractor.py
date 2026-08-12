
# ============================================================
# 3. FEATURE EXTRACTOR
# ============================================================

from app.services.graph_state import GraphState


def feature_extractor(graph: GraphState) -> GraphState:
    """
    Extracts candidate features from the input dataset.

    Responsibilities:
        - Select useful numerical features.
        - Exclude obvious identifiers.
        - Convert datetime information into useful numerical
          representations where appropriate.
        - Prepare the initial feature matrix.

    This node should eventually contain domain-specific feature
    engineering logic.
    """

    df = graph.get("dataframe")

    if df is None or df.empty:
        return {
            "features": None,
            "feature_columns": [],
            "warning": "No valid dataframe available for feature extraction.",
        }

    profile = graph.get("data_profile", {})

    numeric_columns = profile.get(
        "numeric_columns",
        []
    )

    possible_id_columns = set(
        profile.get("possible_id_columns", [])
    )

    feature_columns = [
        column
        for column in numeric_columns
        if column not in possible_id_columns
    ]

    if not feature_columns:
        return {
            "features": None,
            "feature_columns": [],
            "warning": (
                "No suitable numerical features were found "
                "for anomaly detection."
            ),
        }

    X = df[feature_columns].copy()

    return {
        "features": X,
        "feature_columns": feature_columns,
    }
