# ============================================================
# 6. BASELINE TRAINER
# ============================================================

import uuid
from typing import Any

import joblib
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.covariance import EllipticEnvelope

from app.dataset_ingestion import DATA_ROOT
from app.services.feature_transformer import _GLOBAL_GROUP_KEY
from app.services.model_selector import _group_key_to_str
from app.services.graph_state import GraphState

# NOTE on cross-module coupling: _GLOBAL_GROUP_KEY and _group_key_to_str
# are imported from feature_transformer/model_selector despite the
# underscore prefix, because this is now the THIRD module that needs
# identical group-key handling to stay consistent with the other two.
# Worth extracting into a small shared module (e.g.
# app/services/anomaly_grouping_utils.py) as cleanup — flagging rather
# than silently accumulating this as debt.

SCRATCH_ROOT = DATA_ROOT / "_scratch" / "anomaly_runs"

MODEL_CLASSES = {
    "isolation_forest": IsolationForest,
    "lof": LocalOutlierFactor,
    "one_class_svm": OneClassSVM,
    "elliptic_envelope": EllipticEnvelope,
}


def _build_model(model_name: str, params: dict):
    cls = MODEL_CLASSES[model_name]
    if model_name == "lof":
        # novelty=True makes LOF usable in a reusable train -> validate ->
        # predict pipeline — plain LOF only supports fit_predict on the
        # exact training set, with no separate .predict() for new data.
        return cls(novelty=True, **params)
    return cls(**params)


def baseline_trainer(graph: GraphState) -> dict:
    """
    Trains baseline versions of each group's candidate anomaly models
    (from model_selector), fit only on that group's own rows — same
    per-group principle as feature_transformer's scaling and
    model_selector's model choice. Establishes a baseline before
    hyperparameter_optimizer (next node) does more expensive tuning.

    Group membership is re-derived from transformed_features_path itself
    (grouping_columns survive feature_transformer's scaling step
    untouched) rather than looked up from transform_artifacts, which only
    stores each group's scaler/medians, not its row membership.

    Fitted models are NOT put into graph state directly — they aren't
    JSON-serializable, same reasoning as feature_transformer's scalers —
    they're persisted via joblib to baseline_models_path, and only that
    path + a lightweight per-group/per-model status summary travel
    through state.

    A single model failing to fit for one group doesn't take down every
    other model/group already trained — same partial-failure handling
    used for the LLM hallucination case in model_selector.
    """
    transformed_features_path = graph.get("transformed_features_path")
    model_selection = graph.get("model_selection") or {}
    feature_columns = graph.get("feature_columns") or []
    grouping_columns = graph.get("grouping_columns") or []
    run_id = graph.get("feature_run_id")

    if not transformed_features_path or not feature_columns:
        return {
            "baseline_models_path": None,
            "training_summary": {},
            "warning": "No transformed features available for training.",
        }

    if not model_selection:
        return {
            "baseline_models_path": None,
            "training_summary": {},
            "warning": "No model selection available — nothing to train.",
        }

    try:
        df = pd.read_parquet(transformed_features_path)
        if df.empty:
            return {
                "baseline_models_path": None,
                "training_summary": {},
                "warning": "Transformed feature matrix is empty.",
            }

        if grouping_columns:
            group_iter = df.groupby(grouping_columns, dropna=False).groups.items()
        else:
            group_iter = [(_GLOBAL_GROUP_KEY, df.index)]

        trained_models: dict[str, dict[str, Any]] = {}
        training_summary: dict[str, Any] = {}

        for real_group_key, row_index in group_iter:
            str_key = _group_key_to_str(real_group_key)
            selection = model_selection.get(str_key)

            if not selection or not selection.get("candidate_models"):
                training_summary[str_key] = {
                    "row_count": len(row_index),
                    "trained": [],
                    "failed": {},
                    "warning": (selection or {}).get(
                        "warning", "No candidate models for this group."
                    ),
                }
                continue

            X_group = df.loc[row_index, feature_columns]
            group_models: dict[str, Any] = {}
            failed: dict[str, str] = {}

            for model_name in selection["candidate_models"]:
                params = dict(selection["model_configs"].get(model_name, {}))
                try:
                    model = _build_model(model_name, params)
                    model.fit(X_group)
                    group_models[model_name] = model
                except Exception as e:
                    failed[model_name] = str(e)

            if group_models:
                trained_models[str_key] = group_models

            training_summary[str_key] = {
                "row_count": len(row_index),
                "trained": list(group_models.keys()),
                "failed": failed,
            }

        run_id = run_id or uuid.uuid4().hex
        run_dir = SCRATCH_ROOT / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        baseline_models_path = run_dir / "baseline_models.pkl"
        joblib.dump(trained_models, baseline_models_path)

        return {
            "baseline_models_path": str(baseline_models_path),
            "training_summary": training_summary,
        }
    except Exception as e:
        return {"error": str(e)}