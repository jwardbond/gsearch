import networkx as nx

from gsearch.spatial import SpatialGraphIndex, distance_from_node


def run_gsearch(Q: nx.Graph, G: nx.Graph, tol: float, k: int) -> list:

    G_index = SpatialGraphIndex(G)

    anchor_id = _get_anchor_node(Q)

    # Maximum distance from the anchor node to any other node (anchor eccentricity)
    Q_profile = distance_from_node(anchor_id, Q)
    Q_radius = max(d for _, d in Q_profile)

    candidates = _get_candidate_nodes(anchor_id, Q, G)

    # Pruning loop
    for c, _ in candidates:
        G_crop = G_index.make_crop(c, Q_radius + tol)

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
    pass


def _get_anchor_node(Q: nx.Graph) -> int:
    """Get index of anchor node in Q."""
    return next(iter(Q.nodes()))  # Just return first node for now


def _crop_graph(G: nx.Graph, radius: float, anchor_id: int) -> nx.Graph:
    """Crop graph G to a subgraph of nodes within radius of anchor node."""
