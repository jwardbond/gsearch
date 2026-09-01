import math

import networkx as nx
import numpy as np
import pytest

from gsearch.align_nodes import (
    _cross_signs,
    _make_annulus_mask,
    _make_candidates,
    _split_mask_by_handedness,
    get_candidate_matches,
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
        mask = _make_annulus_mask(Q_ARR, G_ARR, Q_ANCHOR, G_ANCHOR, tol=0.5)

        # one row per Q node, one column per G node (anchors included)
        assert mask.shape == (len(Q_ARR), len(G_ARR))

    def test_matches_brute_force(self):
        tol = 0.5

        mask = _make_annulus_mask(Q_ARR, G_ARR, Q_ANCHOR, G_ANCHOR, tol=tol)

        # every entry agrees with an independent loop computation
        assert np.array_equal(
            mask, brute_force_mask(Q_ARR, G_ARR, Q_ANCHOR, G_ANCHOR, tol)
        )

    def test_translation_invariant(self):
        """Shift all of Q by a constant; the function re-aligns the anchor."""
        shifted_Q = Q_ARR + np.array([50.0, -20.0])

        mask = _make_annulus_mask(shifted_Q, G_ARR, Q_ANCHOR, G_ANCHOR, tol=0.5)

        # absolute Q position is meaningless: the mask is unchanged by the shift
        baseline = _make_annulus_mask(Q_ARR, G_ARR, Q_ANCHOR, G_ANCHOR, tol=0.5)
        assert np.array_equal(mask, baseline)

    def test_anchor_row_matches_near_g_anchor(self):
        mask = _make_annulus_mask(Q_ARR, G_ARR, Q_ANCHOR, G_ANCHOR, tol=0.5)

        # the Q anchor sits at distance 0, so its row is True exactly for the
        # G nodes within tol of the G anchor (only the G anchor itself here)
        expected = np.array([True, False, False, False, False])
        assert np.array_equal(mask[Q_ANCHOR], expected)

    def test_boundary_is_inclusive(self):
        """Q[1] is at distance 3, G[3] at distance 2 -> diff 1.0."""
        tol = 1.0

        mask = _make_annulus_mask(Q_ARR, G_ARR, Q_ANCHOR, G_ANCHOR, tol=tol)

        # a pair whose distance difference equals tol exactly is True
        assert mask[1, 3]

    def test_tol_zero_requires_exact_distance(self):
        mask = _make_annulus_mask(Q_ARR, G_ARR, Q_ANCHOR, G_ANCHOR, tol=0.0)

        # only equal-distance pairs survive a zero tolerance
        assert np.array_equal(
            mask, brute_force_mask(Q_ARR, G_ARR, Q_ANCHOR, G_ANCHOR, tol=0.0)
        )
        # the distance-2 and distance-100 columns match no Q node
        assert not mask[:, 3].any()
        assert not mask[:, 4].any()

    def test_nonzero_anchor_indices(self):
        """Use Q[1] and G[1] as anchors (both at distance 3 from index 0)."""
        # they are coincident after alignment, so the index-0 rows/cols must match
        mask = _make_annulus_mask(Q_ARR, G_ARR, 1, 1, tol=0.0)

        # cross-checked against the brute force with the same anchor indices
        assert np.array_equal(
            mask, brute_force_mask(Q_ARR, G_ARR, 1, 1, tol=0.0)
        )
        # the two anchors coincide, so their mutual entry is True
        assert mask[1, 1]

    def test_empty_Q_arr_raises(self):
        empty = np.empty((0, 2))

        # an empty Q has no nodes to align
        with pytest.raises(ValueError, match="Q_arr"):
            _make_annulus_mask(empty, G_ARR, Q_ANCHOR, G_ANCHOR, tol=0.5)

    def test_empty_G_arr_raises(self):
        empty = np.empty((0, 2))

        # an empty G has no nodes to place in the annuli
        with pytest.raises(ValueError, match="G_arr"):
            _make_annulus_mask(Q_ARR, empty, Q_ANCHOR, G_ANCHOR, tol=0.5)


def as_set(mappings: list[dict[int, int]]) -> set[tuple[tuple[int, int], ...]]:
    """Normalize a list of mappings into an order-independent set for comparison."""
    return {tuple(sorted(m.items())) for m in mappings}


class TestMakeCandidates:
    # Distinct ids (not equal to their row/column indices) so the tests verify
    # that results are translated from array positions back to node ids.
    Q_IDS = [10, 20]
    G_IDS = [100, 200, 300]

    def test_single_unique_mapping(self):
        """Each Q row matches exactly one, distinct G column."""
        mask = np.array([[True, False], [False, True]])

        candidates = _make_candidates((mask,), self.Q_IDS, self.G_IDS[:2])

        # only the identity mapping is possible, keyed by node id
        assert candidates == [{10: 100, 20: 200}]

    def test_enumerates_all_injective_combinations(self):
        """Both rows match both columns."""
        mask = np.array([[True, True], [True, True]])

        candidates = _make_candidates((mask,), self.Q_IDS, self.G_IDS[:2])

        # both 1-to-1 assignments appear, and nothing else
        assert as_set(candidates) == as_set([{10: 100, 20: 200}, {10: 200, 20: 100}])
        assert len(candidates) == 2

    def test_rejects_reused_g_index(self):
        """Both rows can only match column 0, forcing a collision."""
        mask = np.array([[True, False], [True, False]])

        candidates = _make_candidates((mask,), self.Q_IDS, self.G_IDS[:2])

        # a G node cannot be assigned to two Q nodes, so no mapping survives
        assert candidates == []

    def test_row_with_no_match_yields_nothing(self):
        """The first row has no True entry."""
        mask = np.array([[False, False], [True, False]])

        candidates = _make_candidates((mask,), self.Q_IDS, self.G_IDS[:2])

        # one unmatchable Q node collapses the whole product to empty
        assert candidates == []

    def test_more_g_than_q_partial_assignment(self):
        """2 Q rows over 3 G columns with overlapping options."""
        mask = np.array([[True, True, False], [False, True, True]])

        candidates = _make_candidates((mask,), self.Q_IDS, self.G_IDS)

        # every injective pick across the two rows, cross-checked by hand
        assert as_set(candidates) == as_set(
            [{10: 100, 20: 200}, {10: 100, 20: 300}, {10: 200, 20: 300}]
        )

    def test_keys_cover_every_q_row(self):
        mask = np.array([[True, False, True]])
        candidates = _make_candidates((mask,), [10], self.G_IDS)

        # a single-row mask maps Q id 10 to each of its matching G ids
        assert candidates == [{10: 100}, {10: 300}]

    def test_combines_every_mask_in_the_tuple(self):
        """A rotation and a reflection mask each contribute their own poses."""
        rotation = np.array([[True, False], [False, True]])
        reflection = np.array([[False, True], [True, False]])

        candidates = _make_candidates(
            (rotation, reflection), self.Q_IDS, self.G_IDS[:2]
        )

        # results from both masks are concatenated in order
        assert candidates == [{10: 100, 20: 200}, {10: 200, 20: 100}]


class TestCrossSigns:
    def test_left_right_and_on_axis(self):
        """Axis (0,0)->(1,0) points +x; points above / below / ahead-on-line."""
        arr = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, -1.0], [2.0, 0.0]])

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
        arr = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        signs = _cross_signs(arr, 0, 1)

        # both endpoints lie on their own axis, so their sign is 0
        assert signs[0] == 0
        assert signs[1] == 0

    def test_collinear_beyond_segment_is_zero(self):
        """Points on the infinite line but outside the i0..i1 segment."""
        arr = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [-3.0, 0.0]])

        signs = _cross_signs(arr, 0, 1)

        # the sign is about the line, not the segment: both read 0
        assert signs[2] == 0
        assert signs[3] == 0

    def test_reversing_axis_flips_signs(self):
        arr = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])
        forward = _cross_signs(arr, 0, 1)

        reversed_ = _cross_signs(arr, 1, 0)

        # reversing the axis direction negates every sign
        assert np.array_equal(reversed_, -forward)

    def test_atol_snaps_near_axis_to_zero(self):
        """Axis along +x; a point just off the line, with a generous atol."""
        arr = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 0.05], [0.5, 0.5]])

        signs = _cross_signs(arr, 0, 1, atol=0.1)

        # a point within atol of the line snaps to 0
        assert signs[2] == 0
        # one clearly off the line keeps its side
        assert signs[3] == 1


# Shared handedness fixtures: the anchor (idx 0) -> alignment (idx 1) axis points
# along +x, with one node on each side. Node signs about that axis are:
#   idx 0 (axis start) -> 0
#   idx 1 (axis end)   -> 0
#   idx 2 (left)       -> +1
#   idx 3 (right)      -> -1
# G repeats the same sign pattern in a far, translated frame.
Q_HANDED = np.array([[0.0, 0.0], [2.0, 0.0], [1.0, 1.0], [1.0, -1.0]])
G_HANDED = np.array(
    [[100.0, 100.0], [102.0, 100.0], [101.0, 101.0], [101.0, 99.0]]
)


class TestSplitMaskByHandedness:
    def test_outputs_keep_input_shape(self):
        mask = np.ones((4, 4), dtype=bool)

        rotation, reflection = _split_mask_by_handedness(
            mask, Q_HANDED, G_HANDED, 0, 1, 0, 1
        )

        # each output is a (len(Q), len(G)) mask like the input
        assert rotation.shape == (4, 4)
        assert reflection.shape == (4, 4)

    def test_same_side_to_rotation_opposite_to_reflection(self):
        """A full mask so every pairing is classified purely by handedness."""
        mask = np.ones((4, 4), dtype=bool)

        rotation, reflection = _split_mask_by_handedness(
            mask, Q_HANDED, G_HANDED, 0, 1, 0, 1
        )

        # the left Q node keeps rotation on the same side, reflects the opposite
        assert rotation[2, 2] and not rotation[2, 3]
        assert reflection[2, 3] and not reflection[2, 2]
        # the right Q node mirrors that pattern
        assert rotation[3, 3] and not rotation[3, 2]
        assert reflection[3, 2] and not reflection[3, 3]

    def test_collinear_pairs_go_to_both_masks(self):
        """Rows/cols 0 and 1 lie on the axis, so their sign product is 0."""
        mask = np.ones((4, 4), dtype=bool)

        rotation, reflection = _split_mask_by_handedness(
            mask, Q_HANDED, G_HANDED, 0, 1, 0, 1
        )

        # a collinear pairing (sign 0) is orientation-agnostic: it joins both masks
        assert rotation[0].all()
        assert reflection[0].all()

    def test_masks_cover_input_overlapping_only_on_collinear(self):
        """A full mask, so the union and overlap are fixed by handedness alone."""
        mask = np.ones((4, 4), dtype=bool)

        rotation, reflection = _split_mask_by_handedness(
            mask, Q_HANDED, G_HANDED, 0, 1, 0, 1
        )

        # together the halves cover every input entry
        assert np.array_equal(rotation | reflection, mask)
        # they overlap exactly on collinear pairings (rows/cols 0 and 1)
        expected_overlap = np.zeros((4, 4), dtype=bool)
        expected_overlap[[0, 1], :] = True
        expected_overlap[:, [0, 1]] = True
        assert np.array_equal(rotation & reflection, expected_overlap)

    def test_never_adds_matches_absent_from_input(self):
        """Keep only a single same-side entry in the input mask."""
        mask = np.zeros((4, 4), dtype=bool)
        mask[2, 2] = True

        rotation, reflection = _split_mask_by_handedness(
            mask, Q_HANDED, G_HANDED, 0, 1, 0, 1
        )

        # the split can only ever remove entries, never introduce them
        assert rotation[2, 2]
        assert rotation.sum() == 1
        assert not reflection.any()


def make_graph(coords, ids) -> nx.Graph:
    """Build a coordinate-only nx.Graph; edges are irrelevant to alignment."""
    g = nx.Graph()
    for node_id, (x, y) in zip(ids, coords):
        g.add_node(node_id, x=float(x), y=float(y))
    return g


# A scalene 3-4-5 right triangle: all pairwise distances differ, so the correct
# correspondence is unique (no symmetry to spawn alternate valid poses). Query ids
# are non-index values, so a returned mapping proves index -> node-id translation.
QUERY_COORDS = np.array([[0.0, 0.0], [3.0, 0.0], [0.0, 4.0]])
QUERY_IDS = [10, 20, 30]
QUERY_ANCHOR = 10


class TestGetCandidateMatches:
    def test_finds_true_correspondence(self):
        """Embed the triangle translated into a far frame, plus a lone decoy."""
        Q = make_graph(QUERY_COORDS, QUERY_IDS)
        g_coords = [[100.0, 100.0], [103.0, 100.0], [100.0, 104.0], [500.0, 500.0]]
        G = make_graph(g_coords, [101, 102, 103, 104])

        matches = get_candidate_matches(Q, G, QUERY_ANCHOR, 101, tol=0.5)

        # the real pose is recovered
        assert {10: 101, 20: 102, 30: 103} in matches
        # the distance-mismatched decoy is never assigned
        assert all(104 not in pose.values() for pose in matches)

    def test_returns_node_ids_not_indices(self):
        """Ids (10.., 101..) are deliberately not positional indices."""
        Q = make_graph(QUERY_COORDS, QUERY_IDS)
        G = make_graph(
            [[100.0, 100.0], [103.0, 100.0], [100.0, 104.0]], [101, 102, 103]
        )

        matches = get_candidate_matches(Q, G, QUERY_ANCHOR, 101, tol=0.5)

        # every mapping is keyed by Q node ids and valued by G node ids
        assert matches
        assert all(set(pose) <= {10, 20, 30} for pose in matches)
        assert all(set(pose.values()) <= {101, 102, 103} for pose in matches)

    def test_invariant_to_rotation(self):
        """Embed the triangle rotated 90 CCW ((x,y) -> (-y,x)) into the far frame."""
        Q = make_graph(QUERY_COORDS, QUERY_IDS)
        G = make_graph(
            [[100.0, 100.0], [100.0, 103.0], [96.0, 100.0]], [101, 102, 103]
        )

        matches = get_candidate_matches(Q, G, QUERY_ANCHOR, 101, tol=0.5)

        # distances and orientation are preserved, so the pose survives
        assert {10: 101, 20: 102, 30: 103} in matches

    def test_finds_reflected_match(self):
        """Embed the triangle mirrored across the axis ((x,y) -> (x,-y))."""
        Q = make_graph(QUERY_COORDS, QUERY_IDS)
        G = make_graph(
            [[100.0, 100.0], [103.0, 100.0], [100.0, 96.0]], [101, 102, 103]
        )

        matches = get_candidate_matches(Q, G, QUERY_ANCHOR, 101, tol=0.5)

        # the reflection half of the handedness split keeps the flipped pose
        assert {10: 101, 20: 102, 30: 103} in matches

    def test_query_larger_than_target_returns_empty(self):
        """Three query nodes cannot each claim a distinct node in a two-node G."""
        Q = make_graph(QUERY_COORDS, QUERY_IDS)
        G = make_graph([[100.0, 100.0], [103.0, 100.0]], [101, 102])

        matches = get_candidate_matches(Q, G, QUERY_ANCHOR, 101, tol=0.5)

        # too few target nodes: no pose is possible
        assert matches == []

    def test_unmatchable_query_node_returns_empty(self):
        """G has the anchor and a distance-3 node but nothing near distance 4."""
        Q = make_graph(QUERY_COORDS, QUERY_IDS)
        G = make_graph(
            [[100.0, 100.0], [103.0, 100.0], [200.0, 100.0]], [101, 102, 103]
        )

        matches = get_candidate_matches(Q, G, QUERY_ANCHOR, 101, tol=0.5)

        # the distance-4 query node has an empty annulus, so nothing matches
        assert matches == []
