import itertools

import networkx as nx
import numpy as np

# TODO I currently compute the distance profile of Q every time. This is constant, so I could optimize.


class NoMatchError(Exception):
    """Raised when there are no possible matches."""


def get_candidate_matches(
    Q: nx.Graph,
    G: nx.Graph,
    Q_anchor: int,
    G_anchor: int,
    tol: float,
) -> list[dict[int, int]]:
    """Get candidate poses by aligning nodes in Q to nodes in G.

    Finds node to node correspondences between Q and G, assuming that the anchor node in
    Q is aligned with the candidate node in G.

    This ignores topology.

    Args:
        Q: The query graph.
        G: The graph to align with Q.
        Q_anchor: The id of the anchor node in Q.
        G_anchor: The id of the candidate node in G to align with Q_anchor.
        tol: The maximum allowed distance between corresponding nodes in Q and G.

    Returns:
        A list of candidate poses, where each pose is a dictionary mapping node IDs in Q to
        node IDs in G.
    """
    # Convert to np arrays
    Q_map = list(Q.nodes)
    G_map = list(G.nodes)

    Q_arr = np.array([[Q.nodes[n]["x"], Q.nodes[n]["y"]] for n in Q_map])
    G_arr = np.array([[G.nodes[n]["x"], G.nodes[n]["y"]] for n in G_map])

    # Every node in Q needs at least 1 node in G
    if len(Q_arr) > len(G_arr):
        return []

    anmsk = make_annulus_mask(
        Q_arr,
        G_arr,
        Q_map.index(Q_anchor),
        G_map.index(G_anchor),
        tol,
    )

    # Every Q vertex needs at least one G vertex in its annulus
    if not anmsk.any(axis=1).all():
        return []

    # Pick the least populated annulus, excluding anchor, and use to align second node
    counts = anmsk.sum(axis=1).astype(float)
    counts[Q_map.index(Q_anchor)] = np.inf
    q_idx = np.argmin(counts)
    q_idx = int(q_idx)

    candidate_matches = []
    for g_idx in np.where(anmsk[q_idx])[0]:
        g_idx = int(g_idx)
        second_anmsk = make_annulus_mask(
            Q_arr,
            G_arr,
            q_idx,
            g_idx,
            tol,
        )

        anmsk_combined = anmsk & second_anmsk
        if not anmsk_combined.any(axis=1).all():
            continue

        candidate_matches.extend(
            make_candidate_combinations(anmsk_combined, Q_map, G_map)
        )

    return candidate_matches


def make_annulus_mask(
    Q_arr: np.ndarray,
    G_arr: np.ndarray,
    Q_anchor_idx: int,
    G_anchor_idx: int,
    tol: float,
) -> np.ndarray:
    """Align Q to G and create annulus masks for each node in Q.

    An annulus mask is the mask created by:
        1. Aligning Q_arr to G_arr by translating the anchor node in Q to the anchor node in G.
        1. Rotating Q_arr around the anchor point, tracing the vertices as they make a circle
        2. Buffering each trace by tol, creating an annulus.
        3. Marking the points within arr2 that fall within each annulus.

    I do that here, just in a clever (if I do say so myself), vectorized way.

    Args:
        Q_arr: Coordinates of the query nodes, shape (len(Q), 2).
        G_arr: Coordinates of the target nodes, shape (len(G), 2).
        Q_anchor_idx: Row index of the anchor node in Q_arr.
        G_anchor_idx: Row index of the candidate anchor node in G_arr.
        tol: The maximum allowed distance between corresponding nodes in Q and G.

    Returns:
        A boolean array of shape (len(Q_arr), len(G_arr)). Each row is a node in Q,
        each column a node in G (anchors included). An entry is True if, after aligning
        the anchors, the distance from the anchor to the Q node is within tol of the
        distance from the anchor to the G node.
    """
    if len(Q_arr) == 0:
        raise ValueError("Q_arr must contain at least one point")
    if len(G_arr) == 0:
        raise ValueError("G_arr must contain at least one point")

    # Transform Q to align with anchor
    Q_pt = Q_arr[Q_anchor_idx]
    G_pt = G_arr[G_anchor_idx]
    transform = G_pt - Q_pt
    Q_arr = Q_arr + transform

    dists1 = np.linalg.norm(Q_arr - G_pt, axis=1)
    dists2 = np.linalg.norm(G_arr - G_pt, axis=1)

    diff_dists = np.abs(dists1[:, None] - dists2[None, :])
    dist_matches = diff_dists <= tol

    return dist_matches


def make_candidate_combinations(
    mask: np.ndarray,
    Q_map: list[int],
    G_map: list[int],
) -> list[dict[int, int]]:
    """Convert a boolean mask of candidate matches into a list of possible combinations.

    When a node in Q has multiple candidate matches in G, this function generates all possible combinations of matches.

    Args:
        mask: A boolean array of shape (len(Q_map), len(G_map)). Each row is a node in Q,
            each column a node in G (anchors included).
        Q_map: List of node IDs in Q corresponding to the rows of mask.
        G_map: List of node IDs in G corresponding to the columns of mask.

    Returns:
        A list of dictionaries, where each dictionary represents a possible combination of candidate matches.

    """
    g_arr = np.array(G_map)
    row_matches = [g_arr[np.flatnonzero(row)] for row in mask]

    return [dict(zip(Q_map, combo)) for combo in itertools.product(*row_matches)]
