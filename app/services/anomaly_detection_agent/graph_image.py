from pathlib import Path

from app.services.anomaly_detection_agent.graph import (
    build_anomaly_graph,
)


def save_anomaly_graph_image() -> None:
    """
    Generates a PNG visualization of the anomaly detection
    subgraph and saves it inside the anomaly_detection_agent
    directory.
    """

    anomaly_graph = build_anomaly_graph()

    png_bytes = anomaly_graph.get_graph().draw_mermaid_png()

    output_path = (
        Path(__file__).resolve().parent
        / "anomaly_agent_graph.png"
    )

    with open(output_path, "wb") as file:
        file.write(png_bytes)

    print(
        f"Saved anomaly graph to:\n{output_path}"
    )


if __name__ == "__main__":
    save_anomaly_graph_image()