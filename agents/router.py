"""Agent 1: Router & Retriever.

Given the user's natural-language question, this agent:

1. Runs a FAISS similarity search over the schema/document index to find the
   most relevant tables and, separately, the most relevant document chunks
   (business glossary, KPI notes, schema commentary, ...).
2. Expands the matched tables to their directly-connected neighbors in the
   NetworkX foreign-key graph, so Agent 2 gets the join path it needs, not
   just the table names it happened to match on.

Output is a single structured, JSON-serializable context payload consumed by
Agent 2 (Text-to-SQL) and Agent 3 (Synthesizer).
"""

import time
import networkx as nx
from utils.faiss_indexer import VectorIndexer


def route_query(query: str, indexer: VectorIndexer, graph: nx.DiGraph, top_k: int = 3):
    trace = []

    start = time.perf_counter()
    schema_hits = indexer.search(query, top_k=top_k, doc_type="schema")
    schema_latency_ms = (time.perf_counter() - start) * 1000
    trace.append({
        "agent": "router.faiss_schema", "status": "ok",
        "latency_ms": round(schema_latency_ms, 1), "hits": len(schema_hits),
    })

    start = time.perf_counter()
    document_hits = indexer.search(query, top_k=top_k, doc_type="document")
    doc_latency_ms = (time.perf_counter() - start) * 1000
    trace.append({
        "agent": "router.faiss_documents", "status": "ok",
        "latency_ms": round(doc_latency_ms, 1), "hits": len(document_hits),
    })

    matched_tables = [d["id"] for d in schema_hits]

    # Expand matched tables to directly-connected neighbors (both directions
    # of the foreign-key edges) so the SQL agent can see the join path.
    start = time.perf_counter()
    subgraph_nodes = set(matched_tables)
    for table in matched_tables:
        if table in graph:
            subgraph_nodes.update(graph.successors(table))
            subgraph_nodes.update(graph.predecessors(table))

    edges = []
    if subgraph_nodes:
        subgraph = graph.subgraph(subgraph_nodes)
        edges = list(subgraph.edges())
    graph_latency_ms = (time.perf_counter() - start) * 1000
    trace.append({
        "agent": "router.graph_traversal", "status": "ok",
        "latency_ms": round(graph_latency_ms, 1), "nodes": len(subgraph_nodes),
    })

    return {
        "tables": matched_tables,
        "schema_context": [d["text"] for d in schema_hits],
        "document_context": [d["text"] for d in document_hits],
        "graph_edges": edges,
        "graph_nodes": sorted(subgraph_nodes),
        "trace": trace,
    }
