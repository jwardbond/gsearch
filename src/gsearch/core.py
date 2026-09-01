import networkx as nx

from gsearch.align_nodes import get_candidate_matches
from gsearch.spatial_graph_index import SpatialGraphIndex


def run_gsearch(Q: nx.Graph, G: nx.Graph, tol: float, k: int) -> list:

    G_index = SpatialGraphIndex(G)

    anchor_id = _get_anchor_node(Q)

    # Maximum distance from the anchor node to any other node (anchor eccentricity)
    Q_profile = _distance_from_node(anchor_id, Q)
    Q_radius = max(d for _, d in Q_profile)

    candidates = _get_candidate_nodes(anchor_id, Q, G)

    # Pruning loop
    for c_id, _ in candidates:
        G_crop = G_index.make_crop(c_id, Q_radius + tol)

        # Get candidate poses by aligning nodes
        poses = get_candidate_poses(Q, G_crop, anchor_id, c_id, tol)

        # Score candidate poses by aligning edges

    # - crop to Q diameter + buffer
    # - discard if not enough nodes
    # - get pairwise distances anchor to nodes in Q
    # - get pairwise distance anchor to nodes in G
    # - for
    # Prune candidates in G by
    # - clipping to Q diameter + buffer
    # - filtering by tolerance


def _get_candidate_nodes(
    anchor_id: int,
    Q: nx.Graph,
    G: nx.Graph,
) -> list[tuple[int, float]]:
    """Get nodes of G sorted by similarity to anchor node in Q."""
    candidates = []
    for node_id in G.nodes():
        score = _similarity_score(anchor_id, node_id, Q, G)
        # Store score with node_id
        candidates.append((node_id, score))

    candidates = sorted(candidates, key=lambda x: x[1], reverse=True)

    return candidates


def _similarity_score(
    anchor_id: int, candidate_id: int, Q: nx.Graph, G: nx.Graph
) -> float:
    return 1.0


def _get_anchor_node(Q: nx.Graph) -> int:
    """Get index of anchor node in Q."""
    return next(iter(Q.nodes()))  # Just return first node for now


def _euclidean_distance(node1: dict, node2: dict) -> float:
    """Calculate Euclidean distance between two nodes."""
    x1, y1 = node1["x"], node1["y"]
    x2, y2 = node2["x"], node2["y"]
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5


def _distance_from_node(anchor_id: int, graph: nx.Graph) -> list[tuple[int, float]]:
    """Get distances from one node to all other nodes in a graph.

    Returns:
        A list of tuples containing (node_id, distance) for each node in the graph,
        excluding the anchor node itself, sorted nearest first.
    """
    distances = []
    for node_id, node_data in graph.nodes(data=True):
        if node_id == anchor_id:
            continue
        distance = _euclidean_distance(graph.nodes[anchor_id], node_data)

        distances.append((node_id, distance))

    distances.sort(key=lambda pair: pair[1])

    return distances
