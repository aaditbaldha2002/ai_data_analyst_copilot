# ============================================================
# 5. MODEL SELECTOR
# ============================================================

import json

import joblib
from openai import OpenAI
from pydantic import BaseModel, Field, ValidationError

from app.config import settings
from app.services.graph_state import GraphState

client = OpenAI(api_key=settings.openai_api_key)

# Currently-supported model types. baseline_trainer needs to know how to
# fit all of these — expand this set (and baseline_trainer) together when
# adding further options like DBSCAN (doesn't use a "contamination"
# parameter the same way — eps/min_samples driven instead, breaks the
# consistent config shape here) or Autoencoders (needs a deep-learning
# framework not yet in this project's stack).
SUPPORTED_MODELS = ("isolation_forest", "lof", "one_class_svm", "elliptic_envelope")

# Hard technical floor, not a judgment call: below this, no model can
# produce a meaningful fit at all. The LLM is never allowed to override
# this regardless of its reasoning.
MIN_ROWS_FOR_ANY_MODEL = 5

DEFAULT_CONTAMINATION = 0.05


class GroupModelSelection(BaseModel):
    # NOTE: deliberately list[str], not list[Literal["isolation_forest", "lof"]].
    # A strict Literal constraint here would make the ENTIRE response fail
    # pydantic validation the moment any single group has an invalid model
    # name — losing every other group's perfectly valid selection along
    # with it. Filtering unsupported names is done post-parse instead (see
    # SUPPORTED_MODELS filtering below), so one bad entry only affects
    # that one group, not the whole batch.
    candidate_models: list[str] = Field(default_factory=list)
    reasoning: str = ""


class ModelSelectionDecision(BaseModel):
    selections: dict[str, GroupModelSelection] = Field(default_factory=dict)


SYSTEM_PROMPT = """You are a senior ML engineer choosing which anomaly-detection algorithms are \
appropriate candidates for each group in a dataset that's being analyzed separately per group.

You are given, for each group: its row count. You are also given the number of feature columns \
(dimensionality), the overall anomaly-detection objective/context, and a target contamination rate.

The models currently available are:
- "isolation_forest": tree-based, robust across group sizes and dimensionality, generally a safe \
  default even for smaller groups.
- "lof" (Local Outlier Factor): density-based, needs a genuinely meaningful local neighborhood to work \
  well — weak on very small groups, and loses effectiveness in higher dimensions (curse of \
  dimensionality). Only include it when the group has enough rows to support real neighborhood density.
- "one_class_svm": kernel-based (RBF), can capture non-linear boundaries between normal and anomalous \
  points. Works on small-to-medium groups but gets computationally slower as group size grows, and its \
  decision boundary can be sensitive to the contamination/nu setting — best suited when you expect a \
  complex, non-linear boundary rather than simple outlier points.
- "elliptic_envelope": assumes the normal data is roughly Gaussian-distributed and fits a robust \
  covariance estimate. Works well when that Gaussian assumption is reasonable and there are CLEARLY \
  more rows than feature columns (needs meaningfully more samples than dimensions to fit a stable \
  covariance matrix) — a poor choice for skewed/multi-modal data or when row count is close to feature \
  count.

Rules:
1. Only choose from "isolation_forest", "lof", "one_class_svm", "elliptic_envelope" — no other model names.
2. A group with very few rows should get an empty or minimal candidate list, not a forced choice.
3. Base your judgment on the actual row counts and dimensionality given, not a fixed rule of thumb.
4. Prefer suggesting 1-3 well-justified candidates per group rather than including every model by default \
   — only include a model if its assumptions genuinely fit this group's size/shape.
5. Give a short (1 sentence) reasoning per group.

Respond with ONLY a JSON object in this exact shape:
{
  "selections": {
    "<group_key>": {"candidate_models": ["isolation_forest", ...], "reasoning": "<short reasoning>"},
    ...
  }
}
One entry per group key given to you — don't invent group keys, don't omit any.
"""


def _group_key_to_str(key) -> str:
    """Groupby keys are tuples when grouping_columns has more than one
    column, scalars when it has one. Normalized to a string for the LLM
    round-trip; mapped back to the original key afterward."""
    if isinstance(key, tuple):
        return "|".join(str(k) for k in key)
    return str(key)


def _ask_llm_for_candidates(
    group_row_counts: dict[str, int],
    feature_count: int,
    contamination: float,
    context: dict,
) -> dict[str, GroupModelSelection]:
    user_prompt = (
        f"Per-group row counts: {json.dumps(group_row_counts)}\n\n"
        f"Number of feature columns (dimensionality): {feature_count}\n\n"
        f"Target contamination rate: {contamination}\n\n"
        f"Anomaly-detection context: {json.dumps(context, default=str)}"
    )

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,
        response_format={"type": "json_object"},
    )

    raw = response.choices[0].message.content
    try:
        parsed = ModelSelectionDecision.model_validate(json.loads(raw))
    except (json.JSONDecodeError, ValidationError) as e:
        raise ValueError(f"Model selection decision returned invalid output: {e}\nRaw response: {raw}")

    return parsed.selections


def _build_config(model_name: str, row_count: int, contamination: float) -> dict:
    if model_name == "isolation_forest":
        return {"contamination": contamination, "n_estimators": 200, "random_state": 42}
    if model_name == "lof":
        return {"n_neighbors": min(20, row_count - 1), "contamination": contamination}
    if model_name == "one_class_svm":
        # nu is OneClassSVM's analog to "contamination" — an upper bound
        # on the fraction of training errors / outliers it expects.
        return {"nu": contamination, "kernel": "rbf", "gamma": "scale"}
    if model_name == "elliptic_envelope":
        return {"contamination": contamination, "random_state": 42}
    return {}


def model_selector(graph: GraphState) -> dict:
    """
    Asks an LLM to decide which anomaly-detection model(s) are appropriate
    candidates for EACH group, given that group's real row count and the
    dataset's dimensionality/context — rather than a fixed row-count
    threshold rule.

    Hard technical floors are still enforced in code regardless of what
    the LLM decides, same principle as every other structured-output step
    in this pipeline: groups below MIN_ROWS_FOR_ANY_MODEL never get a
    model no matter what the LLM says, and LOF is only kept if
    n_neighbors (derived from that group's actual row count) is at least
    1 — scikit-learn's hard requirement, not a preference.

    Numeric hyperparameters (n_estimators, n_neighbors) are computed in
    code, not by the LLM — that's hyperparameter_optimizer's job later in
    the pipeline. This node only decides which model TYPES are worth
    trying per group.
    """
    transform_artifacts_path = graph.get("transform_artifacts_path")
    if not transform_artifacts_path:
        return {
            "model_selection": {},
            "warning": "No transformed features available for model selection.",
        }

    try:
        artifacts = joblib.load(transform_artifacts_path)
    except Exception as e:
        return {"error": f"Could not load transform artifacts: {e}"}

    groups = artifacts.get("groups", {})
    if not groups:
        return {
            "model_selection": {},
            "warning": "No groups found in transform artifacts.",
        }

    try:
        decision = graph.get("profile_decision") or {}
        contamination = decision.get("contamination_hint", DEFAULT_CONTAMINATION)
        if not isinstance(contamination, (int, float)) or not (0 < contamination < 1):
            contamination = DEFAULT_CONTAMINATION

        feature_columns = graph.get("feature_columns") or artifacts.get("feature_columns", [])
        problem = graph.get("anomaly_problem", {}) or {}
        context = {
            "objective": problem.get("objective"),
            "suggested_analysis_dimension": decision.get("suggested_analysis_dimension"),
            "reasoning_from_profiling": decision.get("reasoning"),
        }

        # Key normalization: LLM only ever sees/returns strings.
        key_lookup = {_group_key_to_str(gk): gk for gk in groups.keys()}
        # .get() with a default rather than direct indexing — artifacts
        # come from a pickled file, which has no schema guarantee the way
        # a Pydantic-parsed LLM response does. A malformed/corrupted group
        # entry shouldn't crash the whole node; it just can't be sized,
        # so it's dropped rather than assumed to be a fixed shape.
        group_row_counts = {
            str_key: groups[orig_key].get("row_count")
            for str_key, orig_key in key_lookup.items()
        }
        key_lookup = {
            k: v for k, v in key_lookup.items() if group_row_counts.get(k) is not None
        }

        # Groups below the hard floor are never even offered to the LLM —
        # no ambiguity, no chance of the LLM "deciding" to include one anyway.
        eligible_row_counts = {
            k: v for k, v in group_row_counts.items()
            if k in key_lookup and v >= MIN_ROWS_FOR_ANY_MODEL
        }

        llm_selections = (
            _ask_llm_for_candidates(eligible_row_counts, len(feature_columns), contamination, context)
            if eligible_row_counts
            else {}
        )

        model_selection = {}
        for str_key, orig_key in key_lookup.items():
            row_count = group_row_counts[str_key]

            if row_count < MIN_ROWS_FOR_ANY_MODEL:
                model_selection[str_key] = {
                    "candidate_models": [],
                    "model_configs": {},
                    "row_count": row_count,
                    "warning": "Not enough observations for reliable anomaly detection.",
                }
                continue

            # Hallucination guard: only trust group keys and model names the
            # LLM was actually offered — never trust a fabricated key or a
            # model name outside SUPPORTED_MODELS. Deliberately NOT enforced
            # via a strict pydantic Literal (see GroupModelSelection) since
            # that would fail the entire parsed response over one bad
            # group's entry. Missing keys (LLM omitted a group) fall back
            # to empty rather than erroring.
            selection = llm_selections.get(str_key)
            candidates = list(selection.candidate_models) if selection else []
            candidates = [m for m in candidates if m in SUPPORTED_MODELS]

            # Hard technical floor for LOF specifically, regardless of LLM
            # judgment: n_neighbors must be >= 1, i.e. row_count >= 2. In
            # practice MIN_ROWS_FOR_ANY_MODEL already guarantees this, but
            # kept explicit as defense in depth if that constant ever changes.
            if "lof" in candidates and (row_count - 1) < 1:
                candidates = [m for m in candidates if m != "lof"]

            # Hard technical floor for Elliptic Envelope: its covariance
            # estimate is singular/undefined unless there are strictly more
            # rows than feature columns. A group with 6 rows and 8 features
            # cannot fit this model no matter what the LLM decided.
            if "elliptic_envelope" in candidates and row_count <= len(feature_columns):
                candidates = [m for m in candidates if m != "elliptic_envelope"]

            model_configs = {m: _build_config(m, row_count, contamination) for m in candidates}

            model_selection[str_key] = {
                "candidate_models": candidates,
                "model_configs": model_configs,
                "row_count": row_count,
                "reasoning": selection.reasoning if selection else None,
            }

        validation_strategy = problem.get("validation_strategy", "unsupervised")

        return {
            "model_selection": model_selection,
            "validation_strategy": validation_strategy,
            "contamination_used": contamination,
        }
    except Exception as e:
        return {"error": str(e)}