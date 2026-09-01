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
    # Convert to np array
    Q_ids = list(Q.nodes)
    G_ids = list(G.nodes)

    Q_arr = np.array([[Q.nodes[n]["x"], Q.nodes[n]["y"]] for n in Q_ids])
    G_arr = np.array([[G.nodes[n]["x"], G.nodes[n]["y"]] for n in G_ids])

    Q_anchor_idx = Q_ids.index(Q_anchor)

    # Every node in Q needs at least 1 node in G
    if len(Q_arr) > len(G_arr):
        return []

    anchor_mask = _make_annulus_mask(
        Q_arr,
        G_arr,
        Q_anchor_idx,
        G_ids.index(G_anchor),
        tol,
    )

    # Every Q vertex needs at least one G vertex in its annulus
    if not anchor_mask.any(axis=1).all():
        return []

    # Pick the least populated annulus, excluding anchor, and use to align second node
    counts = anchor_mask.sum(axis=1).astype(float)
    counts[Q_anchor_idx] = np.inf
    second_q_idx = int(np.argmin(counts))

    candidate_matches = []
    for second_g_idx in np.where(anchor_mask[second_q_idx])[0]:
        second_g_idx = int(second_g_idx)
        second_mask = _make_annulus_mask(
            Q_arr,
            G_arr,
            second_q_idx,
            second_g_idx,
            tol,
        )

        combined_mask = anchor_mask & second_mask
        if not combined_mask.any(axis=1).all():
            continue

        candidate_matches.extend(
            _prune_candidates(
                _make_candidates(combined_mask),
                Q_arr,
                G_arr,
                Q_anchor_idx,
                second_q_idx,
            )
        )
    return candidate_matches


def _make_annulus_mask(
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
    offset = G_pt - Q_pt
    Q_arr = Q_arr + offset

    Q_dists = np.linalg.norm(Q_arr - G_pt, axis=1)
    G_dists = np.linalg.norm(G_arr - G_pt, axis=1)

    diff_dists = np.abs(Q_dists[:, None] - G_dists[None, :])
    dist_matches = diff_dists <= tol

    return dist_matches


def _make_candidates(mask: np.ndarray) -> list[dict[int, int]]:
    """Convert a boolean mask into unique 1-to-1 candidate pose mappings.

    Returns:
        A list of dictionaries, where each dictionary maps indices of Q to indices of G.
    """
    row_matches = [np.flatnonzero(row).tolist() for row in mask]

    candidates = []
    n_q = len(mask)
    for combo in itertools.product(*row_matches):
        if len(set(combo)) == n_q:
            candidates.append(dict(enumerate(combo)))

    return candidates


def _prune_candidates(candidate_matches, Q_arr, G_arr, q_axis_a, q_axis_b):
    """Drop poses whose node handedness is not a single global choice.

    A rigid match must preserve orientation consistently. Relative to the axis
    through the two Q anchor nodes (and the matching axis in G), every
    non-collinear node must land on a consistent side: either all sides agree
    (a pure rotation) or all flip (a pure reflection).
    """
    # Q axis is constant across every pose in this batch
    q_signs = _cross_signs(Q_arr, q_axis_a, q_axis_b)

    kept = []
    for pose in candidate_matches:
        q_idx = np.fromiter(pose.keys(), dtype=int, count=len(pose))
        g_idx = np.fromiter(pose.values(), dtype=int, count=len(pose))

        # G axis is the pair matched to the two Q anchors
        g_signs = _cross_signs(G_arr, pose[q_axis_a], pose[q_axis_b])[g_idx]
        q_pose_signs = q_signs[q_idx]

        # Collinear nodes sit on the axis and carry no handedness information
        informative = (q_pose_signs != 0) & (g_signs != 0)
        if not informative.any():
            kept.append(pose)
            continue

        agree = q_pose_signs[informative] == g_signs[informative]
        if agree.all() or (~agree).all():
            kept.append(pose)

    return kept


def _cross_signs(arr, i0, i1, atol=1e-9):
    """Signed side (+1/-1/0) of every point about the directed i0->i1 axis."""
    axis = arr[i1] - arr[i0]
    rel = arr - arr[i0]
    cross = axis[0] * rel[:, 1] - axis[1] * rel[:, 0]
    cross[np.abs(cross) < atol] = 0.0
    return np.sign(cross)
