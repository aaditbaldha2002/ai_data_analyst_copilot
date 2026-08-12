import numpy as np
import pandas as pd

from app.services.graph_state import GraphState

def data_profiler(graph: GraphState) -> GraphState:
    """
    Profiles the input dataset before feature engineering.

    Responsibilities:
        - Determine dataset dimensions.
        - Identify numerical, categorical, and datetime columns.
        - Identify missing values.
        - Identify possible ID columns.
        - Provide metadata for model selection.
    """

    data = graph.get("data", graph.get("result", []))

    if not data:
        return {
            "data_profile": {
                "valid": False,
                "warning": "No data was provided for anomaly detection.",
            }
        }

    df = pd.DataFrame(data)

    numeric_columns = df.select_dtypes(
        include=[np.number]
    ).columns.tolist()

    datetime_columns = df.select_dtypes(
        include=["datetime64[ns]", "datetime64[ns, UTC]"]
    ).columns.tolist()

    categorical_columns = df.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    missing_values = {
        column: int(df[column].isna().sum())
        for column in df.columns
        if df[column].isna().any()
    }

    possible_id_columns = [
        column
        for column in df.columns
        if column.lower().endswith("_id")
        or column.lower() == "id"
    ]

    profile = {
        "valid": True,
        "row_count": len(df),
        "column_count": len(df.columns),
        "columns": df.columns.tolist(),
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "datetime_columns": datetime_columns,
        "missing_values": missing_values,
        "possible_id_columns": possible_id_columns,
    }

    return {
        "dataframe": df,
        "data_profile": profile,
    }
