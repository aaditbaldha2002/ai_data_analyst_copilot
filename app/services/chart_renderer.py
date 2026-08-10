import os
import uuid
import matplotlib
matplotlib.use("Agg")  # non-interactive backend, safe for server use
import matplotlib.pyplot as plt

CHART_DIR = "generated_charts"
os.makedirs(CHART_DIR, exist_ok=True)


def render_chart(chart_config: dict, result: list[dict]) -> str | None:
    chart_type = chart_config.get("chart_type")
    x_key = chart_config.get("x_key")
    y_key = chart_config.get("y_key")
    title = chart_config.get("title") or ""

    if chart_type not in ("bar", "line", "pie") or not x_key or not y_key or not result:
        return None

    try:
        x_values = [str(row.get(x_key)) for row in result]
        y_values = [float(row.get(y_key)) for row in result]
    except (TypeError, ValueError):
        return None

    fig, ax = plt.subplots(figsize=(8, 5))

    if chart_type == "bar":
        ax.bar(x_values, y_values)
        ax.set_xlabel(x_key)
        ax.set_ylabel(y_key)
        plt.xticks(rotation=45, ha="right")
    elif chart_type == "line":
        ax.plot(x_values, y_values, marker="o")
        ax.set_xlabel(x_key)
        ax.set_ylabel(y_key)
        plt.xticks(rotation=45, ha="right")
    elif chart_type == "pie":
        ax.pie(y_values, labels=x_values, autopct="%1.1f%%")

    ax.set_title(title)
    fig.tight_layout()

    filename = f"{uuid.uuid4().hex}.png"
    file_path = os.path.join(CHART_DIR, filename)
    fig.savefig(file_path)
    plt.close(fig)

    return file_path