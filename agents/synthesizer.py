"""Agent 3: Graph Synthesizer & Visualizer.

Formats the final answer for the user: a short deterministic natural-language
summary of the query result (no extra LLM call needed - this keeps the last
step of the pipeline fast and immune to Groq availability issues), plus a
NetworkX plot of the entity subgraph that Agent 1 selected for this query.
"""

import matplotlib.pyplot as plt
import networkx as nx


def generate_graph_plot(graph_edges, highlight_nodes=None):
    """Draws a NetworkX graph plot of the active query relationships.

    Never raises: an empty/None edge list renders a placeholder figure
    instead of crashing the Streamlit page.
    """
    G = nx.DiGraph()

    if graph_edges:
        G.add_edges_from(graph_edges)

    fig, ax = plt.subplots(figsize=(6, 4))

    if len(G.edges) > 0:
        pos = nx.spring_layout(G, seed=42)
        node_colors = "skyblue"
        if highlight_nodes:
            highlight = set(highlight_nodes)
            node_colors = ["orange" if n in highlight else "skyblue" for n in G.nodes]
        nx.draw(
            G, pos,
            with_labels=True,
            node_color=node_colors,
            edge_color='gray',
            node_size=2500,
            font_size=9,
            ax=ax,
        )
        edge_labels = nx.get_edge_attributes(G, "relationship")
        if edge_labels:
            nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7, ax=ax)
    else:
        ax.text(
            0.5, 0.5,
            "No active sub-graph edges found",
            horizontalalignment='center',
            verticalalignment='center',
            transform=ax.transAxes,
        )
        ax.axis('off')

    fig.tight_layout()
    return fig


def summarize_result(user_query: str, sql_query: str, df, profile: dict) -> str:
    """Produces a short, deterministic natural-language summary of the
    query result. Intentionally does not call an LLM - the pipeline's final
    step should not depend on another network call succeeding.
    """
    row_count = profile.get("row_count", len(df))
    columns = profile.get("columns", list(df.columns))

    if row_count == 0:
        return f"Your query ran successfully but returned no rows. Double-check the filters implied by: \"{user_query}\"."

    lines = [f"Found **{row_count}** row(s) across **{len(columns)}** column(s)."]

    null_counts = profile.get("null_counts", {})
    non_zero_nulls = {k: v for k, v in null_counts.items() if v}
    if non_zero_nulls:
        lines.append("Null values detected in: " + ", ".join(f"{k} ({v})" for k, v in non_zero_nulls.items()) + ".")

    try:
        top_row = df.iloc[0].to_dict()
        preview = ", ".join(f"{k}={v}" for k, v in list(top_row.items())[:4])
        lines.append(f"Top result: {preview}.")
    except Exception:
        pass

    return " ".join(lines)
