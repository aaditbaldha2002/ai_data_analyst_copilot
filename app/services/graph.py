from langgraph.graph import StateGraph, END
from app.services.graph_state import GraphState
from app.services.graph_nodes import (
    planner_node, historical_node, forecast_node,
    anomaly_node, root_cause_node, finalize_node,
)

_builder = StateGraph(GraphState)

_builder.add_node("planner", planner_node)
_builder.add_node("historical_task", historical_node)
_builder.add_node("forecast_task", forecast_node)
_builder.add_node("anomaly_task", anomaly_node)
_builder.add_node("root_cause_task", root_cause_node)
_builder.add_node("finalize", finalize_node)

_builder.set_entry_point("planner")

_builder.add_conditional_edges(
    "planner",
    lambda state: state["intent"],
    {
        "historical": "historical_task",
        "forecast": "forecast_task",
        "anomaly": "anomaly_task",
        "root_cause": "root_cause_task",
    },
)

_builder.add_edge("historical_task", "finalize")
_builder.add_edge("forecast_task", "finalize")
_builder.add_edge("anomaly_task", "finalize")
_builder.add_edge("root_cause_task", "finalize")
_builder.add_edge("finalize", END)

copilot_graph = _builder.compile()