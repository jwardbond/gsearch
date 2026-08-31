import networkx as nx
import numpy as np

# TODO I currently compute the distance profile of Q every time. This is constant, so I could optimize.


class NoMatchError(Exception):
    """Raised when there are no possible matches."""


def get_candidate_poses(
    Q: nx.Graph,
    G: nx.Graph,
    Q_anchor: int,
    G_anchor: int,
    tol: float,
) -> list[dict[int, int]]:
    """Get candidate poses by aligning nodes in Q to nodes in G.

    Finds node to node correspondences between Q and G, assuming that the anchor node in
    Q is aligned with the candidate node in G. Guarantees that the distance between pairs
    of nodes in Q and their corresponding nodes in G is within the specified tolerance.

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
    # Make into numpy arrays
    Q_arr = np.array(
        [[Q.nodes[n]["x"], Q.nodes[n]["y"]] for n in Q.nodes() if n != Q_anchor]
    )
    G_arr = np.array(
        [[G.nodes[n]["x"], G.nodes[n]["y"]] for n in G.nodes() if n != G_anchor]
    )
    Q_pt = np.array([Q.nodes[Q_anchor]["x"], Q.nodes[Q_anchor]["y"]])
    G_pt = np.array([G.nodes[G_anchor]["x"], G.nodes[G_anchor]["y"]])

    # Every node in Q needs at least 1 node in G
    if len(Q_arr) > len(G_arr):
        return []

    # Transform Q to align with anchor
    mask = create_annulus_mask(Q_arr, G_arr, Q_pt, tol)

    # Every Q vertex needs at least one G vertex in its annulus
    if not mask.any(axis=1).all():
        return []

    # Smallest supported set
    least_sup_idx = np.argmin(dist_matches.sum(axis=1))

    return []


def create_annulus_mask(
    arr1: np.ndarray,
    arr2: np.ndarray,
    pt: np.ndarray,
    tol: float,
) -> np.ndarray:
    """Create a boolean mask indicating which points in arr2 are within tol of the distances
    from pt to each point in arr1.

    This is the same as doing:
        1. Rotating arr1 around pt, tracing each point.
        2. Buffering each trace by tol, creating an annulus.
        3. Mark the points within arr2 that fall within each annulus.

    Returns:
        A boolean array of shape (len(arr1), len(arr2)). Each row is a point in arr1, each column is a point in arr2.
            An entry is True if the distance from pt to the arr1 point is within tol of the distance from pt to the arr2 point.

    Raises:
        ValueError: If arr1 or arr2 is empty.
    """
    if len(arr1) == 0:
        raise ValueError("arr1 must contain at least one point")
    if len(arr2) == 0:
        raise ValueError("arr2 must contain at least one point")

    dists1 = np.linalg.norm(arr1 - pt, axis=1)
    dists2 = np.linalg.norm(arr2 - pt, axis=1)

    diff_dists = np.abs(dists1[:, None] - dists2[None, :])
    dist_matches = diff_dists <= tol

    return dist_matches
