import json
from pathlib import Path

import networkx as nx


def load_graph(path: Path) -> nx.Graph:
    """Load a graph from a JSON file."""
    with open(path, "r") as f:
        data = json.load(f)

    G = nx.Graph()
    for node in data["nodes"]:
        G.add_node(node["id"], x=node["x"], y=node["y"])
    for edge in data["edges"]:
        G.add_edge(edge["source"], edge["target"])

    return G
