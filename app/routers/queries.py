import os
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.database import get_db
from app.security import get_current_user
from app.models.users import User
from app.models.datasets import Dataset
from app.models.queries import Query
from app.schemas.queries import QueryRequest, QueryResult
from app.services.anomaly_agent import detect_anomalies
from app.services.graph_context import GraphContext
from app.services.sql_agent import (
    generate_sql,
    generate_forecast_sql,
    generate_anomaly_sql,
    generate_root_cause_sql,
)
from app.services.duckdb_engine import run_query
from app.services.chart_agent import generate_chart_config
from app.services.explanation_agent import (
    generate_explanation,
    generate_forecast_explanation,
    generate_anomaly_explanation,
    generate_root_cause_explanation,
)
from app.services.chart_renderer import render_chart
from app.services.intent_agent import classify_intent
from app.services.forecast_agent import generate_forecast, _identify_columns
from app.services.root_cause_agent import analyze_root_cause

router = APIRouter(prefix="/datasets", tags=["queries"])


@router.post("/{dataset_id}/query", response_model=QueryResult)
def query_dataset(
    dataset_id: int,
    request: QueryRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    dataset = (
        db.query(Dataset)
        .filter(Dataset.id == dataset_id, Dataset.owner_id == current_user.id)
        .first()
    )
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    intent = classify_intent(request.question)

    if intent == "forecast":
        sql = generate_forecast_sql(request.question, dataset.schema_json)
    elif intent == "anomaly":
        sql = generate_anomaly_sql(request.question, dataset.schema_json)
    elif intent == "root_cause":
        rc_output = generate_root_cause_sql(request.question, dataset.schema_json)
        sql = rc_output["sql"]
        target_metric = rc_output["target_metric"]
    else:
        sql = generate_sql(request.question, dataset.schema_json)

    error = None
    result = None
    chart_config = None
    chart_image_path = None
    explanation = None
    forecast_model_used = None
    forecast_predictions = None
    anomaly_model_used = None
    anomaly_count = None
    root_cause_analysis = None

    try:
        ext = os.path.splitext(dataset.file_path)[1].lower()
        result = run_query(dataset.file_path, ext, sql)

        if intent == "forecast":
            date_key, value_key = _identify_columns(result)
            forecast = generate_forecast(result, date_key, value_key, periods=3)
            forecast_model_used = forecast["model_used"]
            forecast_predictions = forecast["predictions"]

            combined = result + [
                {date_key: p["date"], value_key: p["predicted_value"], "forecasted": True}
                for p in forecast_predictions
            ]
            chart_config = generate_chart_config(request.question, combined)
            chart_config["chart_type"] = "line"
            chart_image_path = render_chart(chart_config, combined)
            explanation = generate_forecast_explanation(
                request.question, result, forecast_predictions, forecast_model_used
            )

        elif intent == "anomaly":
            anomaly_result = detect_anomalies(result)
            result = anomaly_result["rows"]
            anomaly_model_used = anomaly_result["model_used"]
            anomaly_count = anomaly_result["anomaly_count"]

            chart_config = generate_chart_config(request.question, result)
            chart_config["chart_type"] = "scatter"
            chart_image_path = render_chart(chart_config, result)
            explanation = generate_anomaly_explanation(request.question, result)

        elif intent == "root_cause":
            analysis = analyze_root_cause(result, target_metric)
            root_cause_analysis = analysis
            explanation = generate_root_cause_explanation(request.question, analysis)
            chart_config = None
            chart_image_path = None
            
        else:
            chart_config = generate_chart_config(request.question, result)
            chart_image_path = render_chart(chart_config, result)
            explanation = generate_explanation(request.question, result)

    except Exception as e:
        error = str(e)

    query_record = Query(
        dataset_id=dataset.id,
        owner_id=current_user.id,
        question=request.question,
        generated_sql=sql,
        result_json=result,
        chart_config=chart_config,
        chart_image_path=chart_image_path,
        explanation=explanation,
        forecast_model_used=forecast_model_used,
        forecast_predictions=forecast_predictions,
        anomaly_model_used=anomaly_model_used,
        anomaly_count=anomaly_count,
        root_cause_analysis=root_cause_analysis,
        error=error,
    )
    db.add(query_record)
    db.commit()
    db.refresh(query_record)

    if error:
        raise HTTPException(status_code=400, detail={"sql": sql, "error": error})

    return query_record

from app.services.graph import copilot_graph

@router.post("/{dataset_id}/query-v2", response_model=QueryResult)
def query_dataset_v2(
    request: QueryRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    initial_state = {
        "question": request.question,
        "owner_id": current_user.id,
    }

    final_state = copilot_graph.invoke(initial_state,context=GraphContext(db=db))

    query_record = Query(
        dataset_id=final_state.get("dataset_id"),
        owner_id=current_user.id,
        question=request.question,
        generated_sql=final_state.get("sql", ""),
        result_json=final_state.get("result"),
        chart_config=final_state.get("chart_config"),
        chart_image_path=final_state.get("chart_image_path"),
        explanation=final_state.get("explanation"),
        forecast_model_used=final_state.get("forecast_model_used"),
        forecast_predictions=final_state.get("forecast_predictions"),
        anomaly_model_used=final_state.get("anomaly_model_used"),
        anomaly_count=final_state.get("anomaly_count"),
        root_cause_analysis=final_state.get("root_cause_analysis"),
        error=final_state.get("error"),
    )
    db.add(query_record)
    db.commit()
    db.refresh(query_record)

    if final_state.get("error"):
        raise HTTPException(status_code=400, detail={"sql": final_state.get("sql"), "error": final_state["error"]})

    return query_record