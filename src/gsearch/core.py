import logging

import networkx as nx

from gsearch.align_edges import align_and_score
from gsearch.align_nodes import get_candidate_matches
from gsearch.spatial_graph_index import SpatialGraphIndex

logger = logging.getLogger(__name__)


def run_gsearch(Q: nx.Graph, G: nx.Graph, tol: float, k: int) -> list:

    G_index = SpatialGraphIndex(G)

    Q_anchor_id = _get_anchor_node(Q)
    Q_radius = _get_radius(Q, Q_anchor_id)  # Max distance from anchor to any other node

    # Build list of candidate anchors in G
    candidate_anchors = _get_candidate_anchors(Q_anchor_id, Q, G)
    logger.debug("Found %d candidate anchors", len(candidate_anchors))

    # Create possible matches around anchor, align, and score
    alignments = []
    for G_anchor_id, _ in candidate_anchors:
        G_crop = G_index.make_crop(G_anchor_id, Q_radius + tol)

        logger.debug("get_candidate_matches called for anchor %s", G_anchor_id)
        matches = get_candidate_matches(Q, G_crop, Q_anchor_id, G_anchor_id, tol)
        logger.debug(
            "get_candidate_matches done for anchor %s: %d matches",
            G_anchor_id,
            len(matches),
        )

        for match in matches:
            result = align_and_score(Q, G_crop, match, tol)

            if result is None:
                continue

            alignments.append(result)

    alignments = sorted(alignments, key=lambda x: x[0], reverse=True)

    k = min(k, len(alignments))
    return alignments[:k]


def _get_radius(Q: nx.Graph, anchor_id: int) -> float:
    """Get the radius of the query graph Q, defined as the maximum distance from the anchor node to any other node."""
    distances = _distance_from_node(anchor_id, Q)
    radius = max(d for _, d in distances)
    return radius


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


def _euclidean_distance(node1: dict, node2: dict) -> float:
    """Calculate Euclidean distance between two nodes."""
    x1, y1 = node1["x"], node1["y"]
    x2, y2 = node2["x"], node2["y"]
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5


def _get_candidate_anchors(
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
