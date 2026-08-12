from langgraph.graph import StateGraph, START, END

from app.services.anomaly_detection_agent.model_evaluater import model_evaluator
from app.services.anomaly_detection_agent.problem_analyzer import anomaly_problem_analyzer
from app.services.graph_state import GraphState


from app.services.anomaly_detection_agent.data_profiler import (
    data_profiler,
)

from app.services.anomaly_detection_agent.feature_extractor import (
    feature_extractor,
)

from app.services.anomaly_detection_agent.feature_transformer import (
    feature_transformer,
)

from app.services.anomaly_detection_agent.model_selector import (
    model_selector,
)

from app.services.anomaly_detection_agent.baseline_trainer import (
    baseline_trainer,
)


from app.services.anomaly_detection_agent.hyperparameter_optimizer import (
    hyperparameter_optimizer,
)

from app.services.anomaly_detection_agent.final_model_trainer import (
    final_model_trainer,
)

from app.services.anomaly_detection_agent.anomaly_scorer import (
    anomaly_scorer,
)

from app.services.anomaly_detection_agent.anomaly_explainer import (
    anomaly_explainer,
)

from app.services.anomaly_detection_agent.response_generator import (
    response_generator,
)


def build_anomaly_graph():
    """
    Builds and compiles the anomaly detection subgraph.

    This graph is mounted into the master Copilot graph
    under the 'anomaly_task' node.
    """

    builder = StateGraph(GraphState)

    # ========================================================
    # Nodes
    # ========================================================

    builder.add_node(
        "anomaly_problem_analyzer",
        anomaly_problem_analyzer,
    )

    builder.add_node(
        "data_profiler",
        data_profiler,
    )

    builder.add_node(
        "feature_extractor",
        feature_extractor,
    )

    builder.add_node(
        "feature_transformer",
        feature_transformer,
    )

    builder.add_node(
        "model_selector",
        model_selector,
    )

    builder.add_node(
        "baseline_trainer",
        baseline_trainer,
    )

    builder.add_node(
        "model_evaluator",
        model_evaluator,
    )

    builder.add_node(
        "hyperparameter_optimizer",
        hyperparameter_optimizer,
    )

    builder.add_node(
        "final_model_trainer",
        final_model_trainer,
    )

    builder.add_node(
        "anomaly_scorer",
        anomaly_scorer,
    )

    builder.add_node(
        "anomaly_explainer",
        anomaly_explainer,
    )

    builder.add_node(
        "response_generator",
        response_generator,
    )

    # ========================================================
    # Workflow
    # ========================================================

    builder.add_edge(
        START,
        "anomaly_problem_analyzer",
    )

    builder.add_edge(
        "anomaly_problem_analyzer",
        "data_profiler",
    )

    builder.add_edge(
        "data_profiler",
        "feature_extractor",
    )

    builder.add_edge(
        "feature_extractor",
        "feature_transformer",
    )

    builder.add_edge(
        "feature_transformer",
        "model_selector",
    )

    builder.add_edge(
        "model_selector",
        "baseline_trainer",
    )

    builder.add_edge(
        "baseline_trainer",
        "model_evaluator",
    )

    builder.add_edge(
        "model_evaluator",
        "hyperparameter_optimizer",
    )

    builder.add_edge(
        "hyperparameter_optimizer",
        "final_model_trainer",
    )

    builder.add_edge(
        "final_model_trainer",
        "anomaly_scorer",
    )

    builder.add_edge(
        "anomaly_scorer",
        "anomaly_explainer",
    )

    builder.add_edge(
        "anomaly_explainer",
        "response_generator",
    )

    builder.add_edge(
        "response_generator",
        END,
    )

    return builder.compile()