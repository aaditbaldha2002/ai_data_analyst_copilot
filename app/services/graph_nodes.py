from typing_extensions import runtime

from app.services.graph import GraphContext
from app.services.intent_agent import classify_intent
from app.services.sql_agent import generate_sql, generate_forecast_sql, generate_anomaly_sql, generate_root_cause_sql
from app.services.duckdb_engine import run_query
from app.services.forecast_agent import generate_forecast, _identify_columns
from app.services.anomaly_agent import detect_anomalies
from app.services.root_cause_agent import analyze_root_cause
from app.services.chart_agent import generate_chart_config
from app.services.chart_renderer import render_chart
from app.services.explanation_agent import (
    generate_explanation, generate_forecast_explanation,
    generate_anomaly_explanation, generate_root_cause_explanation,
)
from app.services.graph_state import GraphState
from langgraph.runtime import Runtime

def planner_node(state: GraphState,runtime: Runtime[GraphContext],) -> dict:
    classification = classify_intent(
            question=state["question"],
            owner_id=state["owner_id"],
            db=runtime.context.db,
        )

    return {
        "intent": classification["intent"],
        "dataset_id": classification["dataset_id"],
    }

def _run(state: GraphState, sql: str) -> list[dict]:
    return run_query(state["file_path"], state["ext"], sql)


def historical_node(state: GraphState) -> dict:
    try:
        sql = generate_sql(state["question"], state["schema"])
        result = _run(state, sql)
        return {"sql": sql, "result": result}
    except Exception as e:
        return {"error": str(e)}


def forecast_node(state: GraphState) -> dict:
    try:
        sql = generate_forecast_sql(state["question"], state["schema"])
        result = _run(state, sql)
        date_key, value_key = _identify_columns(result)
        forecast = generate_forecast(result, date_key, value_key, periods=3)
        return {
            "sql": sql,
            "result": result,
            "forecast_model_used": forecast["model_used"],
            "forecast_predictions": forecast["predictions"],
        }
    except Exception as e:
        return {"error": str(e)}


def anomaly_node(state: GraphState) -> dict:
    try:
        sql = generate_anomaly_sql(state["question"], state["schema"])
        result = _run(state, sql)
        anomaly_result = detect_anomalies(result)
        return {
            "sql": sql,
            "result": anomaly_result["rows"],
            "anomaly_model_used": anomaly_result["model_used"],
            "anomaly_count": anomaly_result["anomaly_count"],
        }
    except Exception as e:
        return {"error": str(e)}


def root_cause_node(state: GraphState) -> dict:
    try:
        rc_output = generate_root_cause_sql(state["question"], state["schema"])
        sql = rc_output["sql"]
        target_metric = rc_output["target_metric"]
        result = _run(state, sql)
        analysis = analyze_root_cause(result, target_metric)
        return {
            "sql": sql,
            "result": result,
            "target_metric": target_metric,
            "root_cause_analysis": analysis,
        }
    except Exception as e:
        return {"error": str(e)}


def finalize_node(state: GraphState) -> dict:
    if state.get("error"):
        return {}

    intent = state["intent"]
    question = state["question"]
    result = state.get("result") or []

    if intent == "forecast":
        date_key, value_key = _identify_columns(result)
        combined = result + [
            {date_key: p["date"], value_key: p["predicted_value"], "forecasted": True}
            for p in state.get("forecast_predictions", [])
        ]
        chart_config = generate_chart_config(question, combined)
        chart_config["chart_type"] = "line"
        chart_image_path = render_chart(chart_config, combined)
        explanation = generate_forecast_explanation(
            question, result, state.get("forecast_predictions", []), state.get("forecast_model_used")
        )
        return {"chart_config": chart_config, "chart_image_path": chart_image_path, "explanation": explanation}

    if intent == "anomaly":
        chart_config = generate_chart_config(question, result)
        chart_config["chart_type"] = "scatter"
        chart_image_path = render_chart(chart_config, result)
        explanation = generate_anomaly_explanation(question, result)
        return {"chart_config": chart_config, "chart_image_path": chart_image_path, "explanation": explanation}

    if intent == "root_cause":
        explanation = generate_root_cause_explanation(question, state.get("root_cause_analysis", {}))
        return {"chart_config": None, "chart_image_path": None, "explanation": explanation}

    chart_config = generate_chart_config(question, result)
    chart_image_path = render_chart(chart_config, result)
    explanation = generate_explanation(question, result)
    return {"chart_config": chart_config, "chart_image_path": chart_image_path, "explanation": explanation}