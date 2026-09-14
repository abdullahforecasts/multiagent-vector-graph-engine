import sqlite3
import networkx as nx

def build_knowledge_graph(db_path: str = "data/chinook.db") -> nx.DiGraph:
    """Extracts table schema and foreign keys into a NetworkX directed graph."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    G = nx.DiGraph()

    # Get all table names
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [row[0] for row in cursor.fetchall() if not row[0].startswith("sqlite_")]

    for table in tables:
        G.add_node(table, type="table")
        
        # Get foreign key relationships
        cursor.execute(f"PRAGMA foreign_key_list('{table}');")
        for fk in cursor.fetchall():
            target_table = fk[2]
            from_col = fk[3]
            to_col = fk[4]
            G.add_edge(
                table, 
                target_table, 
                relationship="foreign_key", 
                from_col=from_col, 
                to_col=to_col
            )

    conn.close()
    return G