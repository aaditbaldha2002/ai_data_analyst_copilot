import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler


def _get_numeric_columns(df: pd.DataFrame) -> list[str]:
    return df.select_dtypes(include=[np.number]).columns.tolist()


def detect_anomalies(result: list[dict], contamination: float = 0.1) -> dict:
    """
    Runs anomaly detection on the numeric columns of the result set.
    Returns rows annotated with anomaly flags/scores, plus metadata on which model(s) ran.
    """
    df = pd.DataFrame(result)

    # Convert any datetime columns to plain strings so the result is JSON-serializable
    # (pandas Timestamp objects are not JSON serializable and will break the DB insert).
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = df[col].astype(str)

    numeric_cols = _get_numeric_columns(df)

    if not numeric_cols or len(df) < 5:
        return {
            "model_used": None,
            "rows": result,
            "anomaly_count": 0,
            "warning": "Not enough numeric data to run anomaly detection.",
        }

    X = df[numeric_cols].fillna(df[numeric_cols].median())
    X_scaled = StandardScaler().fit_transform(X)

    n = len(df)
    use_lof = n >= 20

    # --- Isolation Forest (always run as the baseline) ---
    iso = IsolationForest(contamination=contamination, random_state=42)
    iso_labels = iso.fit_predict(X_scaled)          # -1 = anomaly, 1 = normal
    iso_scores = iso.decision_function(X_scaled)     # lower = more anomalous

    df["_iso_anomaly"] = iso_labels == -1
    df["_iso_score"] = iso_scores

    if use_lof:
        lof = LocalOutlierFactor(n_neighbors=min(20, n - 1), contamination=contamination)
        lof_labels = lof.fit_predict(X_scaled)
        df["_lof_anomaly"] = lof_labels == -1
        # consensus: flagged by either model, but note strength if flagged by both
        df["is_anomaly"] = df["_iso_anomaly"] | df["_lof_anomaly"]
        df["anomaly_confidence"] = df["_iso_anomaly"].astype(int) + df["_lof_anomaly"].astype(int)
        model_used = "isolation_forest+lof"
        df = df.drop(columns=["_lof_anomaly"])
    else:
        df["is_anomaly"] = df["_iso_anomaly"]
        df["anomaly_confidence"] = df["_iso_anomaly"].astype(int)
        model_used = "isolation_forest"

    df = df.drop(columns=["_iso_anomaly", "_iso_score"])

    anomaly_count = int(df["is_anomaly"].sum())

    return {
        "model_used": model_used,
        "rows": df.to_dict(orient="records"),
        "anomaly_count": anomaly_count,
        "warning": None,
    }