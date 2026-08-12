import os
import uuid
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

REPORT_DIR = "generated_reports"
os.makedirs(REPORT_DIR, exist_ok=True)


def generate_dashboard_report(
    dataset_name: str,
    kpis: dict,
    segmentation: dict,
    chart_paths: list[str],
) -> str:
    filename = f"{uuid.uuid4().hex}.pdf"
    file_path = os.path.join(REPORT_DIR, filename)

    doc = SimpleDocTemplate(file_path, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph(f"Data Analyst Copilot — Dashboard Report", styles["Title"]))
    story.append(Paragraph(f"Dataset: {dataset_name}", styles["Normal"]))
    story.append(Spacer(1, 0.3 * inch))

    # KPI table
    story.append(Paragraph("Key Metrics", styles["Heading2"]))
    kpi_rows = [[k.replace("_", " ").title(), str(v)] for k, v in kpis.items()]
    kpi_table = Table([["Metric", "Value"]] + kpi_rows, colWidths=[3 * inch, 2 * inch])
    kpi_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2c3e50")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
    ]))
    story.append(kpi_table)
    story.append(Spacer(1, 0.3 * inch))

    # Charts
    for path in chart_paths:
        if path and os.path.exists(path):
            story.append(Image(path, width=5.5 * inch, height=3.4 * inch))
            story.append(Spacer(1, 0.2 * inch))

    # Segmentation
    if segmentation.get("segments"):
        story.append(Paragraph("Segmentation", styles["Heading2"]))
        for seg in segmentation["segments"]:
            label = seg["segment"]
            members = seg.get(segmentation["group_column"], [])
            story.append(Paragraph(f"<b>{label}:</b> {', '.join(str(m) for m in members)}", styles["Normal"]))
            story.append(Spacer(1, 0.1 * inch))

    doc.build(story)
    return file_path