import math

import numpy as np
import pytest

from gsearch.align_nodes import (
    _cross_signs,
    _make_annulus_mask,
    _make_candidates,
    _prune_candidates,
)

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
        mask = _make_annulus_mask(Q_ARR, G_ARR, Q_ANCHOR, G_ANCHOR, tol=0.5)

        # one row per Q node, one column per G node (anchors included)
        assert mask.shape == (len(Q_ARR), len(G_ARR))

    def test_matches_brute_force(self):
        # plan
        tol = 0.5

        # do
        mask = _make_annulus_mask(Q_ARR, G_ARR, Q_ANCHOR, G_ANCHOR, tol=tol)

        # every entry agrees with an independent loop computation
        assert np.array_equal(
            mask, brute_force_mask(Q_ARR, G_ARR, Q_ANCHOR, G_ANCHOR, tol)
        )

    def test_translation_invariant(self):
        # plan: shift all of Q by a constant; the function re-aligns the anchor
        shifted_Q = Q_ARR + np.array([50.0, -20.0])

        # do
        mask = _make_annulus_mask(shifted_Q, G_ARR, Q_ANCHOR, G_ANCHOR, tol=0.5)

        # absolute Q position is meaningless: the mask is unchanged by the shift
        baseline = _make_annulus_mask(Q_ARR, G_ARR, Q_ANCHOR, G_ANCHOR, tol=0.5)
        assert np.array_equal(mask, baseline)

    def test_anchor_row_matches_near_g_anchor(self):
        # plan / do
        mask = _make_annulus_mask(Q_ARR, G_ARR, Q_ANCHOR, G_ANCHOR, tol=0.5)

        # the Q anchor sits at distance 0, so its row is True exactly for the
        # G nodes within tol of the G anchor (only the G anchor itself here)
        expected = np.array([True, False, False, False, False])
        assert np.array_equal(mask[Q_ANCHOR], expected)

    def test_boundary_is_inclusive(self):
        # plan: Q[1] is at distance 3, G[3] at distance 2 -> diff 1.0
        tol = 1.0

        # do
        mask = _make_annulus_mask(Q_ARR, G_ARR, Q_ANCHOR, G_ANCHOR, tol=tol)

        # a pair whose distance difference equals tol exactly is True
        assert mask[1, 3]

    def test_tol_zero_requires_exact_distance(self):
        # plan / do
        mask = _make_annulus_mask(Q_ARR, G_ARR, Q_ANCHOR, G_ANCHOR, tol=0.0)

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
        mask = _make_annulus_mask(Q_ARR, G_ARR, 1, 1, tol=0.0)

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
            _make_annulus_mask(empty, G_ARR, Q_ANCHOR, G_ANCHOR, tol=0.5)

    def test_empty_G_arr_raises(self):
        # plan
        empty = np.empty((0, 2))

        # do / test: an empty G has no nodes to place in the annuli
        with pytest.raises(ValueError, match="G_arr"):
            _make_annulus_mask(Q_ARR, empty, Q_ANCHOR, G_ANCHOR, tol=0.5)


def as_set(mappings: list[dict[int, int]]) -> set[tuple[tuple[int, int], ...]]:
    """Normalize a list of mappings into an order-independent set for comparison."""
    return {tuple(sorted(m.items())) for m in mappings}


class TestMakeCandidates:
    def test_single_unique_mapping(self):
        # plan: each Q row matches exactly one, distinct G column
        mask = np.array([[True, False], [False, True]])

        # do
        candidates = _make_candidates(mask)

        # only the identity mapping is possible
        assert candidates == [{0: 0, 1: 1}]

    def test_enumerates_all_injective_combinations(self):
        # plan: both rows match both columns
        mask = np.array([[True, True], [True, True]])

        # do
        candidates = _make_candidates(mask)

        # both 1-to-1 assignments appear, and nothing else
        assert as_set(candidates) == as_set([{0: 0, 1: 1}, {0: 1, 1: 0}])
        assert len(candidates) == 2

    def test_rejects_reused_g_index(self):
        # plan: both rows can only match column 0, forcing a collision
        mask = np.array([[True, False], [True, False]])

        # do
        candidates = _make_candidates(mask)

        # a G node cannot be assigned to two Q nodes, so no mapping survives
        assert candidates == []

    def test_row_with_no_match_yields_nothing(self):
        # plan: the first row has no True entry
        mask = np.array([[False, False], [True, False]])

        # do
        candidates = _make_candidates(mask)

        # one unmatchable Q node collapses the whole product to empty
        assert candidates == []

    def test_more_g_than_q_partial_assignment(self):
        # plan: 2 Q rows over 3 G columns with overlapping options
        mask = np.array([[True, True, False], [False, True, True]])

        # do
        candidates = _make_candidates(mask)

        # every injective pick across the two rows, cross-checked by hand
        assert as_set(candidates) == as_set(
            [{0: 0, 1: 1}, {0: 0, 1: 2}, {0: 1, 1: 2}]
        )

    def test_keys_cover_every_q_row(self):
        # plan / do
        mask = np.array([[True, False, True]])
        candidates = _make_candidates(mask)

        # a single-row mask maps Q index 0 to each of its matching G columns
        assert candidates == [{0: 0}, {0: 2}]


class TestCrossSigns:
    def test_left_right_and_on_axis(self):
        # plan: axis (0,0)->(1,0) points +x; points above / below / ahead-on-line
        arr = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, -1.0], [2.0, 0.0]])

        # do
        signs = _cross_signs(arr, 0, 1)

        # one sign per input point
        assert signs.shape == (len(arr),)
        # a point left of the directed axis (CCW) is +1
        assert signs[2] == 1
        # a point right of it (CW) is -1
        assert signs[3] == -1
        # a point on the line (ahead of the segment) is 0
        assert signs[4] == 0

    def test_axis_endpoints_are_zero(self):
        # plan / do
        arr = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        signs = _cross_signs(arr, 0, 1)

        # both endpoints lie on their own axis, so their sign is 0
        assert signs[0] == 0
        assert signs[1] == 0

    def test_collinear_beyond_segment_is_zero(self):
        # plan: points on the infinite line but outside the i0..i1 segment
        arr = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [-3.0, 0.0]])

        # do
        signs = _cross_signs(arr, 0, 1)

        # the sign is about the line, not the segment: both read 0
        assert signs[2] == 0
        assert signs[3] == 0

    def test_reversing_axis_flips_signs(self):
        # plan
        arr = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])
        forward = _cross_signs(arr, 0, 1)

        # do
        reversed_ = _cross_signs(arr, 1, 0)

        # reversing the axis direction negates every sign
        assert np.array_equal(reversed_, -forward)

    def test_atol_snaps_near_axis_to_zero(self):
        # plan: axis along +x; a point just off the line, with a generous atol
        arr = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 0.05], [0.5, 0.5]])

        # do
        signs = _cross_signs(arr, 0, 1, atol=0.1)

        # a point within atol of the line snaps to 0
        assert signs[2] == 0
        # one clearly off the line keeps its side
        assert signs[3] == 1


# Shared query for the prune tests: anchors at indices 0 and 1 define an axis
# along +x, with one node on each side. Node signs about that axis are:
#   Q0 (0, 0)   axis start -> 0
#   Q1 (2, 0)   axis end   -> 0
#   Q2 (1, 1)   left       -> +1
#   Q3 (1, -1)  right      -> -1
Q_SIGNED = np.array([[0.0, 0.0], [2.0, 0.0], [1.0, 1.0], [1.0, -1.0]])
FULL_POSE = {0: 0, 1: 1, 2: 2, 3: 3}


class TestPruneCandidates:
    def test_keeps_pure_rotation(self):
        # plan: G is Q rotated 90 CCW into a far frame -> orientation preserved
        G_arr = np.array(
            [[100.0, 100.0], [100.0, 102.0], [99.0, 101.0], [101.0, 101.0]]
        )

        # do
        kept = _prune_candidates([FULL_POSE], Q_SIGNED, G_arr, 0, 1)

        # every node keeps its side, so the pose survives
        assert kept == [FULL_POSE]

    def test_keeps_pure_reflection(self):
        # plan: G is Q mirrored across the axis -> every side flips uniformly
        G_arr = np.array(
            [[100.0, 100.0], [102.0, 100.0], [101.0, 99.0], [101.0, 101.0]]
        )

        # do
        kept = _prune_candidates([FULL_POSE], Q_SIGNED, G_arr, 0, 1)

        # a uniform reflection is still a single global choice, so it is kept
        assert kept == [FULL_POSE]

    def test_drops_mixed_handedness(self):
        # plan: both matched G nodes sit on the +side, so one agrees and one flips
        G_arr = np.array([[0.0, 0.0], [2.0, 0.0], [1.0, 1.0], [1.0, 2.0]])

        # do
        kept = _prune_candidates([FULL_POSE], Q_SIGNED, G_arr, 0, 1)

        # inconsistent handedness is geometrically impossible, so the pose is dropped
        assert kept == []

    def test_keeps_when_all_collinear(self):
        # plan: a Q whose non-anchor nodes lie on the axis -> no handedness info
        Q_collinear = np.array([[0.0, 0.0], [2.0, 0.0], [1.0, 0.0], [0.5, 0.0]])
        G_arr = np.array([[0.0, 0.0], [2.0, 0.0], [1.0, 1.0], [1.0, 2.0]])

        # do
        kept = _prune_candidates([FULL_POSE], Q_collinear, G_arr, 0, 1)

        # with no informative nodes there is nothing to reject, so it is kept
        assert kept == [FULL_POSE]

    def test_empty_input_returns_empty(self):
        # plan / do
        kept = _prune_candidates([], Q_SIGNED, Q_SIGNED, 0, 1)

        # no poses in, no poses out
        assert kept == []

    def test_filters_within_one_batch(self):
        # plan: G holds a rotation set (0-3) and a mixed set (4-7)
        G_arr = np.array(
            [
                [100.0, 100.0],
                [100.0, 102.0],
                [99.0, 101.0],
                [101.0, 101.0],
                [0.0, 0.0],
                [2.0, 0.0],
                [1.0, 1.0],
                [1.0, 2.0],
            ]
        )
        rotation_pose = {0: 0, 1: 1, 2: 2, 3: 3}
        mixed_pose = {0: 4, 1: 5, 2: 6, 3: 7}

        # do
        kept = _prune_candidates(
            [rotation_pose, mixed_pose], Q_SIGNED, G_arr, 0, 1
        )

        # only the consistent pose survives, and its position is preserved
        assert kept == [rotation_pose]
