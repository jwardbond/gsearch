import logging
import math
from collections.abc import Iterator

import networkx as nx
import numpy as np

logger = logging.getLogger(__name__)

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
    logger.debug(
        "get_candidate_matches: |Q|=%d |G|=%d Q_anchor=%s G_anchor=%s tol=%s",
        Q.number_of_nodes(),
        G.number_of_nodes(),
        Q_anchor,
        G_anchor,
        tol,
    )

    # Convert to np array
    Q_ids = list(Q.nodes)
    G_ids = list(G.nodes)

    Q_arr = np.array([[Q.nodes[n]["x"], Q.nodes[n]["y"]] for n in Q_ids])
    G_arr = np.array([[G.nodes[n]["x"], G.nodes[n]["y"]] for n in G_ids])

    Q_anchor_idx = Q_ids.index(Q_anchor)
    G_anchor_idx = G_ids.index(G_anchor)

    # Every node in Q needs at least 1 node in G
    if len(Q_arr) > len(G_arr):
        logger.debug(
            "No matches: |Q|=%d > |G|=%d", len(Q_arr), len(G_arr)
        )
        return []

    anchor_mask = _make_annulus_mask(
        Q_arr,
        G_arr,
        Q_anchor_idx,
        G_anchor_idx,
        tol,
    )

    # Every Q vertex needs at least one G vertex in its annulus
    if not anchor_mask.any(axis=1).all():
        empty_rows = np.flatnonzero(~anchor_mask.any(axis=1)).tolist()
        logger.debug(
            "No matches: Q rows with empty anchor annulus: %s", empty_rows
        )
        return []

    # Pick the least populated annulus, excluding the anchor, and use it to align the
    # alignment node.
    counts = anchor_mask.sum(axis=1).astype(float)
    counts[Q_anchor_idx] = np.inf
    Q_alignment_idx = int(np.argmin(counts))
    logger.debug(
        "Alignment node: Q_alignment_idx=%d with %d G candidates",
        Q_alignment_idx,
        int(counts[Q_alignment_idx]),
    )

    candidate_matches = []
    G_alignment_candidates = np.where(anchor_mask[Q_alignment_idx])[0]
    for i, G_alignment_idx in enumerate(G_alignment_candidates):
        G_alignment_idx = int(G_alignment_idx)
        logger.debug(
            "Alignment loop %d/%d: G_alignment_idx=%d",
            i + 1,
            len(G_alignment_candidates),
            G_alignment_idx,
        )
        alignment_mask = _make_annulus_mask(
            Q_arr,
            G_arr,
            Q_alignment_idx,
            G_alignment_idx,
            tol,
        )

        combined_mask = anchor_mask & alignment_mask

        split_masks = _split_mask_by_handedness(
            combined_mask,
            Q_arr,
            G_arr,
            Q_anchor_idx,
            Q_alignment_idx,
            G_anchor_idx,
            G_alignment_idx,
        )

        new_matches = _make_candidates(split_masks, Q_ids, G_ids)
        logger.debug(
            "G_alignment_idx=%d yielded %d candidate poses",
            G_alignment_idx,
            len(new_matches),
        )
        candidate_matches.extend(new_matches)

    logger.debug("get_candidate_matches: %d total candidate poses", len(candidate_matches))
    return candidate_matches


def _make_annulus_mask(
    Q_arr: np.ndarray,
    G_arr: np.ndarray,
    Q_ref_idx: int,
    G_ref_idx: int,
    tol: float,
) -> np.ndarray:
    """Align Q to G and create annulus masks for each node in Q.

    An annulus mask is the mask created by:
        1. Aligning Q_arr to G_arr by translating the reference node in Q to the reference node in G.
        1. Rotating Q_arr around the reference point, tracing the vertices as they make a circle
        2. Buffering each trace by tol, creating an annulus.
        3. Marking the points within arr2 that fall within each annulus.

    I do that here, just in a clever (if I do say so myself), vectorized way.

    Args:
        Q_arr: Coordinates of the query nodes, shape (len(Q), 2).
        G_arr: Coordinates of the target nodes, shape (len(G), 2).
        Q_ref_idx: Row index of the reference node in Q_arr (anchor or alignment node).
        G_ref_idx: Row index of the candidate reference node in G_arr.
        tol: The maximum allowed distance between corresponding nodes in Q and G.

    Returns:
        A boolean array of shape (len(Q_arr), len(G_arr)). Each row is a node in Q,
        each column a node in G (reference nodes included). An entry is True if, after
        aligning the reference nodes, the distance from the reference to the Q node is
        within tol of the distance from the reference to the G node.
    """
    if len(Q_arr) == 0:
        raise ValueError("Q_arr must contain at least one point")
    if len(G_arr) == 0:
        raise ValueError("G_arr must contain at least one point")

    # Distances are translation-invariant, so aligning the reference nodes is implicit:
    # measure each node's distance from its own reference node.
    Q_ref_pt = Q_arr[Q_ref_idx]
    G_ref_pt = G_arr[G_ref_idx]

    Q_dists = np.linalg.norm(Q_arr - Q_ref_pt, axis=1)
    G_dists = np.linalg.norm(G_arr - G_ref_pt, axis=1)

    diff_dists = np.abs(Q_dists[:, None] - G_dists[None, :])
    dist_matches = diff_dists <= tol

    return dist_matches


def _make_candidates(
    masks: tuple[np.ndarray, ...],
    Q_ids: list[int],
    G_ids: list[int],
) -> list[dict[int, int]]:
    """Convert boolean masks into unique 1-to-1 candidate pose mappings.

    Args:
        masks: Boolean arrays of shape (len(Q), len(G)); each row a Q node, each column a G node.
        Q_ids: Node ids of Q, indexed by row.
        G_ids: Node ids of G, indexed by column.

    Returns:
        A list of dictionaries, where each dictionary maps node ids in Q to node ids in G.
    """
    candidates = []
    for mask_idx, mask in enumerate(masks):
        if not mask.any(axis=1).all():
            continue

        row_matches = [np.flatnonzero(row).tolist() for row in mask]
        product_size = math.prod(len(r) for r in row_matches)

        n_before = len(candidates)
        for combo in _injective_assignments(row_matches):
            candidates.append(
                {Q_ids[qi]: G_ids[gi] for qi, gi in enumerate(combo)},
            )
        logger.debug(
            "_make_candidates: mask %d, per-row match counts %s, "
            "%d injective poses (full product would be %d)",
            mask_idx,
            [len(r) for r in row_matches],
            len(candidates) - n_before,
            product_size,
        )

    return candidates


def _injective_assignments(
    row_matches: list[list[int]],
) -> Iterator[tuple[int, ...]]:
    """Yield every injective assignment of one G index per row, no G index reused.

    Backtracking equivalent of filtering ``itertools.product(*row_matches)`` for
    combos with all-distinct entries, but it abandons a partial assignment as soon
    as it reuses a G index instead of enumerating the full Cartesian product first.

    Args:
        row_matches: For each Q row, the list of candidate G column indices.

    Yields:
        Tuples of G column indices, one per row, with no index repeated.
    """
    n_q = len(row_matches)
    combo: list[int] = []
    used: set[int] = set()

    def backtrack(i: int):
        if i == n_q:
            yield tuple(combo)
            return
        for g in row_matches[i]:
            if g in used:
                continue
            used.add(g)
            combo.append(g)
            yield from backtrack(i + 1)
            combo.pop()
            used.discard(g)

    yield from backtrack(0)


def _split_mask_by_handedness(
    mask: np.ndarray,
    Q_arr: np.ndarray,
    G_arr: np.ndarray,
    Q_anchor_idx: int,
    Q_alignment_idx: int,
    G_anchor_idx: int,
    G_alignment_idx: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Split into (rotation_mask, reflection_mask) by orientation about the axes."""
    Q_signs = _cross_signs(Q_arr, Q_anchor_idx, Q_alignment_idx)  # (n_q,)
    G_signs = _cross_signs(G_arr, G_anchor_idx, G_alignment_idx)  # (n_g,)

    products = Q_signs[:, None] * G_signs[None, :]
    # Collinear nodes (sign 0, always including the anchor and alignment nodes) are
    # orientation-agnostic, so they belong to both masks: same side -> rotation,
    # opposite side -> reflection, on-axis -> both.
    same = products >= 0
    flip = products <= 0
    return mask & same, mask & flip


def _cross_signs(
    arr: np.ndarray,
    i0: int,
    i1: int,
    atol: float = 1e-9,
) -> np.ndarray:
    """Signed side (+1/-1/0) of every point about the directed i0->i1 axis."""
    axis = arr[i1] - arr[i0]
    rel = arr - arr[i0]
    cross = axis[0] * rel[:, 1] - axis[1] * rel[:, 0]
    cross[np.abs(cross) < atol] = 0.0
    return np.sign(cross)
