# ============================================================
# 9. FINAL MODEL TRAINER
# ============================================================

import uuid
from typing import Any

import joblib
import pandas as pd

from app.dataset_ingestion import DATA_ROOT
from app.services.anomaly_detection_agent.feature_transformer import _GLOBAL_GROUP_KEY
from app.services.anomaly_detection_agent.model_selector import _group_key_to_str
from app.services.anomaly_detection_agent.baseline_trainer import _build_model
from app.services.graph_state import GraphState

# Same cross-module coupling flagged in baseline_trainer.py/model_evaluator.py
# — now the fifth module needing identical group-key handling (plus reuse
# of baseline_trainer's _build_model, to avoid a second copy of the
# model-class mapping drifting out of sync with the first). Worth
# extracting into a shared module; flagged again rather than silently
# repeated a third time.

SCRATCH_ROOT = DATA_ROOT / "_scratch" / "anomaly_runs"


def final_model_trainer(graph: GraphState) -> dict:
    """
    Trains the FINAL anomaly detection model(s) per group, using
    hyperparameter_optimizer's optimized_params rather than
    model_selector's baseline configs.

    The current implementation keeps every optimized candidate per group
    — a future node (anomaly_scorer, per your diagram, or a dedicated
    selection step) can choose one final model or construct an ensemble
    from whatever's trained here.

    A (group, model) pair with no entry in optimized_params is skipped
    rather than falling back to some other config — if
    hyperparameter_optimizer had no evaluation feedback to optimize from
    (e.g. that model failed evaluation earlier), there's nothing valid to
    train a "final" version from.

    Same persistence pattern as baseline_trainer: fitted models are never
    put into graph state directly (not JSON-serializable) — persisted via
    joblib to final_models_path, with only that path + a lightweight
    per-group training summary traveling through state. A single model
    failing to fit for one group doesn't take down any other group/model.
    """
    transformed_features_path = graph.get("transformed_features_path")
    optimized_params = graph.get("optimized_params") or {}
    feature_columns = graph.get("feature_columns") or []
    grouping_columns = graph.get("grouping_columns") or []
    run_id = graph.get("feature_run_id")

    if not transformed_features_path or not feature_columns:
        return {
            "final_models_path": None,
            "final_training_summary": {},
            "warning": "No transformed features available for final training.",
        }

    if not optimized_params:
        return {
            "final_models_path": None,
            "final_training_summary": {},
            "warning": "No optimized params available — nothing to train.",
        }

    try:
        df = pd.read_parquet(transformed_features_path)
        if df.empty:
            return {
                "final_models_path": None,
                "final_training_summary": {},
                "warning": "Transformed feature matrix is empty.",
            }

        if grouping_columns:
            group_iter = df.groupby(grouping_columns, dropna=False).groups.items()
        else:
            group_iter = [(_GLOBAL_GROUP_KEY, df.index)]

        final_models: dict[str, dict[str, Any]] = {}
        final_training_summary: dict[str, Any] = {}

        for real_group_key, row_index in group_iter:
            str_key = _group_key_to_str(real_group_key)
            group_params = optimized_params.get(str_key)

            if not group_params:
                final_training_summary[str_key] = {
                    "row_count": len(row_index),
                    "trained": [],
                    "failed": {},
                    "warning": "No optimized params for this group.",
                }
                continue

            X_group = df.loc[row_index, feature_columns]
            group_models: dict[str, Any] = {}
            failed: dict[str, str] = {}

            for model_name, params in group_params.items():
                try:
                    model = _build_model(model_name, dict(params))
                    model.fit(X_group)
                    group_models[model_name] = model
                except Exception as e:
                    failed[model_name] = str(e)

            if group_models:
                final_models[str_key] = group_models

            final_training_summary[str_key] = {
                "row_count": len(row_index),
                "trained": list(group_models.keys()),
                "failed": failed,
            }

        run_id = run_id or uuid.uuid4().hex
        run_dir = SCRATCH_ROOT / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        final_models_path = run_dir / "final_models.pkl"
        joblib.dump(final_models, final_models_path)

        return {
            "final_models_path": str(final_models_path),
            "final_training_summary": final_training_summary,
        }
    except Exception as e:
        return {"error": str(e)}