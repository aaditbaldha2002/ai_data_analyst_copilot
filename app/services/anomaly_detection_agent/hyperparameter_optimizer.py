# ============================================================
# 8. HYPERPARAMETER OPTIMIZER
# ============================================================

from app.services.graph_state import GraphState

# A gap this large between a model's CONFIGURED contamination/nu and its
# OBSERVED anomaly rate is meaningful enough to warrant nudging the
# parameter toward what the data actually showed, rather than leaving a
# config in place that's demonstrably off. Below this, the baseline
# config is left untouched.
DEVIATION_THRESHOLD = 0.05

# Same bounds data_profiler's contamination_hint already uses — never
# push contamination/nu outside a sane range even if the observed rate
# was more extreme than that (e.g. a degenerate small-group fit).
CONTAMINATION_MIN, CONTAMINATION_MAX = 0.01, 0.3


def hyperparameter_optimizer(graph: GraphState) -> dict:
    """
    Lightweight optimization pass over each group's evaluated models —
    NOT a real hyperparameter search. This intentionally stays simple:

    1. Where a model's observed anomaly rate deviated meaningfully from
       its configured contamination/nu target (per model_evaluator's own
       deviation_from_expected diagnostic), nudge that parameter toward
       the observed rate rather than leaving a demonstrably-off config.
    2. A couple of modest, clearly-labeled per-model-type adjustments
       (more estimators for Isolation Forest, a wider neighborhood for
       LOF) — grounded in that GROUP's actual row count, not the whole
       dataset's.

    A production version should replace this with Optuna (or similar)
    once a proper validation objective exists — per the original
    docstring's own note, optimizing against F1/accuracy isn't valid
    without reliable ground-truth labels, and self-evaluation deviation
    is a reasonable but limited proxy in the meantime.

    Only (group, model) pairs that were actually evaluated successfully
    are optimized — a model that failed evaluation has no feedback to
    adjust from, so it's skipped rather than guessed at.
    """
    model_selection = graph.get("model_selection") or {}
    evaluation_results = graph.get("evaluation_results") or {}

    if not model_selection or not evaluation_results:
        return {
            "optimized_params": {},
            "warning": "No model selection or evaluation results available for optimization.",
        }

    try:
        optimized_params: dict[str, dict] = {}

        for group_key, group_eval in evaluation_results.items():
            selection = model_selection.get(group_key, {})
            base_configs = selection.get("model_configs", {})
            row_count = selection.get("row_count")

            group_optimized: dict[str, dict] = {}

            for model_name, eval_metrics in group_eval.items():
                if "error" in eval_metrics:
                    continue  # nothing to optimize from if evaluation itself failed

                base_config = dict(base_configs.get(model_name, {}))
                if not base_config:
                    continue

                observed = eval_metrics.get("anomaly_rate")
                deviation = eval_metrics.get("deviation_from_expected")

                contamination_key = "nu" if model_name == "one_class_svm" else "contamination"
                if (
                    deviation is not None
                    and observed is not None
                    and deviation > DEVIATION_THRESHOLD
                    and contamination_key in base_config
                ):
                    nudged = max(CONTAMINATION_MIN, min(CONTAMINATION_MAX, observed))
                    base_config[contamination_key] = round(nudged, 4)

                if model_name == "isolation_forest":
                    base_config["n_estimators"] = max(base_config.get("n_estimators", 200), 300)
                elif model_name == "lof" and row_count:
                    base_config["n_neighbors"] = min(30, row_count - 1)

                group_optimized[model_name] = base_config

            if group_optimized:
                optimized_params[group_key] = group_optimized

        return {"optimized_params": optimized_params}
    except Exception as e:
        return {"error": str(e)}