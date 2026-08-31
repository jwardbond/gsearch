import math

import numpy as np
import pytest

from gsearch.align_nodes import make_annulus_mask

# Q is centred near the origin, G at large positive coords (as in the real data),
# so absolute position is meaningless: the mask must come only from each node's
# distance to its own anchor after the anchors are aligned.
#
#   Q_arr (rows)    dist from Q anchor (0, 0)
#   (0.0, 0.0)       0.0    <- Q anchor (index 0)
#   (3.0, 0.0)       3.0
#   (0.0, 4.0)       4.0
#
#   G_arr (cols)    dist from G anchor (100, 100)
#   (100.0, 100.0)   0.0    <- G anchor (index 0)
#   (103.0, 100.0)   3.0    <- matches Q[1]
#   (100.0, 104.0)   4.0    <- matches Q[2]
#   (100.0, 102.0)   2.0
#   (200.0, 100.0)  100.0   <- far from every Q distance
Q_ARR = np.array([[0.0, 0.0], [3.0, 0.0], [0.0, 4.0]])
G_ARR = np.array(
    [[100.0, 100.0], [103.0, 100.0], [100.0, 104.0], [100.0, 102.0], [200.0, 100.0]]
)
Q_ANCHOR = 0
G_ANCHOR = 0


def brute_force_mask(
    Q_arr: np.ndarray,
    G_arr: np.ndarray,
    Q_anchor_idx: int,
    G_anchor_idx: int,
    tol: float,
) -> np.ndarray:
    """Annulus mask computed with plain Python loops, no broadcasting."""
    qax, qay = Q_arr[Q_anchor_idx]
    gax, gay = G_arr[G_anchor_idx]
    mask = np.zeros((len(Q_arr), len(G_arr)), dtype=bool)
    for i, (qx, qy) in enumerate(Q_arr):
        d1 = math.hypot(qx - qax, qy - qay)
        for j, (gx, gy) in enumerate(G_arr):
            d2 = math.hypot(gx - gax, gy - gay)
            mask[i, j] = abs(d1 - d2) <= tol
    return mask


class TestMakeAnnulusMask:
    def test_output_shape(self):
        # plan / do
        mask = make_annulus_mask(Q_ARR, G_ARR, Q_ANCHOR, G_ANCHOR, tol=0.5)

        # one row per Q node, one column per G node (anchors included)
        assert mask.shape == (len(Q_ARR), len(G_ARR))

    def test_matches_brute_force(self):
        # plan
        tol = 0.5

        # do
        mask = make_annulus_mask(Q_ARR, G_ARR, Q_ANCHOR, G_ANCHOR, tol=tol)

        # every entry agrees with an independent loop computation
        assert np.array_equal(
            mask, brute_force_mask(Q_ARR, G_ARR, Q_ANCHOR, G_ANCHOR, tol)
        )

    def test_translation_invariant(self):
        # plan: shift all of Q by a constant; the function re-aligns the anchor
        shifted_Q = Q_ARR + np.array([50.0, -20.0])

        # do
        mask = make_annulus_mask(shifted_Q, G_ARR, Q_ANCHOR, G_ANCHOR, tol=0.5)

        # absolute Q position is meaningless: the mask is unchanged by the shift
        baseline = make_annulus_mask(Q_ARR, G_ARR, Q_ANCHOR, G_ANCHOR, tol=0.5)
        assert np.array_equal(mask, baseline)

    def test_anchor_row_matches_near_g_anchor(self):
        # plan / do
        mask = make_annulus_mask(Q_ARR, G_ARR, Q_ANCHOR, G_ANCHOR, tol=0.5)

        # the Q anchor sits at distance 0, so its row is True exactly for the
        # G nodes within tol of the G anchor (only the G anchor itself here)
        expected = np.array([True, False, False, False, False])
        assert np.array_equal(mask[Q_ANCHOR], expected)

    def test_boundary_is_inclusive(self):
        # plan: Q[1] is at distance 3, G[3] at distance 2 -> diff 1.0
        tol = 1.0

        # do
        mask = make_annulus_mask(Q_ARR, G_ARR, Q_ANCHOR, G_ANCHOR, tol=tol)

        # a pair whose distance difference equals tol exactly is True
        assert mask[1, 3]

    def test_tol_zero_requires_exact_distance(self):
        # plan / do
        mask = make_annulus_mask(Q_ARR, G_ARR, Q_ANCHOR, G_ANCHOR, tol=0.0)

        # only equal-distance pairs survive a zero tolerance
        assert np.array_equal(
            mask, brute_force_mask(Q_ARR, G_ARR, Q_ANCHOR, G_ANCHOR, tol=0.0)
        )
        # the distance-2 and distance-100 columns match no Q node
        assert not mask[:, 3].any()
        assert not mask[:, 4].any()

    def test_nonzero_anchor_indices(self):
        # plan: use Q[1] and G[1] as anchors (both at distance 3 from index 0),
        # which are coincident after alignment, so index-0 rows/cols must match
        mask = make_annulus_mask(Q_ARR, G_ARR, 1, 1, tol=0.0)

        # cross-checked against the brute force with the same anchor indices
        assert np.array_equal(
            mask, brute_force_mask(Q_ARR, G_ARR, 1, 1, tol=0.0)
        )
        # the two anchors coincide, so their mutual entry is True
        assert mask[1, 1]

    def test_empty_Q_arr_raises(self):
        # plan
        empty = np.empty((0, 2))

        # do / test: an empty Q has no nodes to align
        with pytest.raises(ValueError, match="Q_arr"):
            make_annulus_mask(empty, G_ARR, Q_ANCHOR, G_ANCHOR, tol=0.5)

    def test_empty_G_arr_raises(self):
        # plan
        empty = np.empty((0, 2))

        # do / test: an empty G has no nodes to place in the annuli
        with pytest.raises(ValueError, match="G_arr"):
            make_annulus_mask(Q_ARR, empty, Q_ANCHOR, G_ANCHOR, tol=0.5)
