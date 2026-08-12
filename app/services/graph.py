from langgraph.graph import StateGraph, START, END

from app.services.anomaly_detection_agent.graph import build_anomaly_graph
from app.services.graph_state import GraphState
from app.services.graph_nodes import (
    planner_node,
    historical_node,
    forecast_node,
    root_cause_node,
    finalize_node,
)



# ============================================================
# SUBGRAPHS
# ============================================================

anomaly_graph = build_anomaly_graph()


# ============================================================
# MASTER GRAPH
# ============================================================

_builder = StateGraph(GraphState)


# ------------------------------------------------------------
# Master-level nodes
# ------------------------------------------------------------

_builder.add_node(
    "planner",
    planner_node,
)

_builder.add_node(
    "historical_task",
    historical_node,
)

_builder.add_node(
    "forecast_task",
    forecast_node,
)

_builder.add_node(
    "anomaly_task",
    anomaly_graph,
)

_builder.add_node(
    "root_cause_task",
    root_cause_node,
)

_builder.add_node(
    "finalize",
    finalize_node,
)


# ============================================================
# ENTRY POINT
# ============================================================

_builder.add_edge(
    START,
    "planner",
)


# ============================================================
# INTENT ROUTING
# ============================================================

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


# ============================================================
# TASK → FINALIZE
# ============================================================

_builder.add_edge(
    "historical_task",
    "finalize",
)

_builder.add_edge(
    "forecast_task",
    "finalize",
)

_builder.add_edge(
    "anomaly_task",
    "finalize",
)

_builder.add_edge(
    "root_cause_task",
    "finalize",
)


# ============================================================
# FINALIZE → END
# ============================================================

_builder.add_edge(
    "finalize",
    END,
)


# ============================================================
# COMPILE MASTER GRAPH
# ============================================================

copilot_graph = _builder.compile()