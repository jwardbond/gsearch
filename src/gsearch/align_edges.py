from itertools import pairwise

import networkx as nx
import numpy as np
from scipy.linalg import svd


def align_and_score(
    Q: nx.Graph,
    G_crop: nx.Graph,
    match: dict[int, int],
    tol: float,
    weight: float = 0.5,
) -> tuple[float, np.ndarray, np.ndarray, nx.Graph, dict[int, int]] | None:
    """Align a candidate match and score how well it fits.

    Aligns Q's node coords onto the matched G nodes (Procrustes), then routes
    each Q edge along its best path through G_crop.

    Args:
        weight: Blends the two residuals into the final score,
            weight * node_error + (1 - weight) * edge_cost.

    Returns:
        A tuple of:
          - The final score.
          - The rotation matrix.
          - The translation vector.
          - The aligned subgraph.
          - The match dictionary.
    """
    Q_nodes = list(Q.nodes)
    G_nodes = list(G_crop.nodes)

    Q_arr = np.array([[Q.nodes[n]["x"], Q.nodes[n]["y"]] for n in Q_nodes])
    G_arr = np.array([[G_crop.nodes[n]["x"], G_crop.nodes[n]["y"]] for n in G_nodes])
    G_arr_matched = G_arr[[G_nodes.index(match[q]) for q in Q_nodes]]

    # Align Q nodes with matched G nodes
    R, t, Q_arr_aligned = _align_graph(Q_arr, G_arr_matched)
    node_score = np.linalg.norm(Q_arr_aligned - G_arr_matched)

    # Get best paths
    result = _find_paths(Q, G_crop, Q_nodes, G_nodes, Q_arr_aligned, G_arr, match, tol)
    if result is None:
        return None
    edge_score, paths = result

    # Assemble final subgraph
    matched_nodes = set(match.values())
    matched_edges: set[tuple[int, int]] = set()
    for path in paths.values():
        matched_nodes.update(path)
        matched_edges.update(pairwise(path))

    subgraph = nx.Graph()
    subgraph.add_nodes_from((n, G_crop.nodes[n]) for n in matched_nodes)
    subgraph.add_edges_from(matched_edges)

    # Total residual
    score = node_score + weight * edge_score

    return score, R, t, subgraph, match


def _align_graph(
    Q_arr: np.ndarray, G_arr: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Align Q onto G with SVD-based Procrustes (Sabata et al. 1991).

    Returns:
        A tuple of:
        - R: The rotation matrix.
        - t: The translation vector.
        - Q_arr_aligned: Q_arr after alignment.

    """
    # Paired coords, one row per match in the same order in both arrays

    # Center coords
    mu_Q = Q_arr.mean(axis=0)
    mu_G = G_arr.mean(axis=0)

    Q_centered = Q_arr - mu_Q
    G_centered = G_arr - mu_G

    # SVD
    H = G_centered.T @ Q_centered
    U, _, Vt = svd(H)
    R = Vt.T @ U.T  # rotation mat (reflections allowed)
    t = mu_G - mu_Q @ R  # translation

    # Apply the transform to all of Q and rebuild the graph
    Q_arr_aligned = Q_arr @ R + t

    return R, t, Q_arr_aligned


def _find_paths(
    Q: nx.Graph,
    G: nx.Graph,
    Q_nodes: list,
    G_nodes: list,
    Q_arr: np.ndarray,  # aligned Q coords, row i == Q_nodes[i]
    G_arr: np.ndarray,  # G_crop coords, row i == G_nodes[i]
    match: dict[int, int],
    tol: float,
) -> tuple[float, dict[tuple, list[int]]] | None:
    """
    Find the best paths in G for each edge in Q.

    Returns:
        A tuple of:
        - edge_score: The total cost of all edges in Q when routed through G.
        - paths: A dictionary mapping each edge in Q to its corresponding path in G.
    """
    Q_pos = {n: Q_arr[i] for i, n in enumerate(Q_nodes)}  # node id -> row in Q_arr
    G_idx = {n: i for i, n in enumerate(G_nodes)}  # node id -> row index in G_arr

    edge_score = 0.0
    paths: dict[tuple, list[int]] = {}

    for u, v in Q.edges:
        u_xy, v_xy = Q_pos[u], Q_pos[v]
        dists = _dist_to_segment(G_arr, u_xy, v_xy)  # per-node distance to segment

        # Callback. Weight edge a-b by the average dist of its endpoints to u_xy, v_xy segment.
        def weight(a, b, _attrs, dists=dists):
            da, db = dists[G_idx[a]], dists[G_idx[b]]
            if da > tol or db > tol:
                return None
            seg_len = np.linalg.norm(G_arr[G_idx[a]] - G_arr[G_idx[b]])
            return 0.5 * (da + db) * seg_len

        try:
            # target is set, so this returns a scalar length and a single path
            length, path = nx.single_source_dijkstra(
                G,
                source=match[u],
                target=match[v],
                weight=weight,
            )
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None

        edge_score += length  # type: ignore
        paths[(u, v)] = path  # type: ignore

    return edge_score, paths


def _dist_to_segment(
    coords: np.ndarray,
    pA: np.ndarray,
    pB: np.ndarray,
) -> np.ndarray:
    """Get the distance from each row of coords (N,2) to the segment pA->pB."""
    AB = pB - pA
    denom = AB @ AB

    # Check if pA==pB
    if denom == 0.0:
        return np.linalg.norm(coords - pA, axis=1)

    t = np.clip((coords - pA) @ AB / denom, 0.0, 1.0)  # (N,)
    proj = pA + t[:, None] * AB  # (N,2)
    return np.linalg.norm(coords - proj, axis=1)
