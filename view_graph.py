from app.services.graph import copilot_graph

png_bytes = copilot_graph.get_graph().draw_mermaid_png()
with open("graph.png", "wb") as f:
    f.write(png_bytes)

print("Saved graph.png")