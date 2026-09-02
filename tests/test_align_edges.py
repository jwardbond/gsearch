import math

import networkx as nx
import numpy as np
import pytest

from gsearch.align_edges import _align_graph, _dist_to_segment, _find_paths

# Q lives near the origin, G in a far, translated frame, as in the real data.
# The rows are a scalene 3-4-5 triangle: all pairwise distances differ, so the
# rigid transform between paired point sets is unique (no symmetry to admit an
# alternate fit). Q_arr and G_arr are already paired row-for-row.
Q_ARR = np.array([[0.0, 0.0], [3.0, 0.0], [0.0, 4.0]])


def rotation(theta: float) -> np.ndarray:
    """A proper 2x2 rotation matrix (det +1) for angle theta in radians."""
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s], [s, c]])


# A known orthonormal transform applied to Q to build a noise-free G. Because the
# points are exact and non-collinear, Procrustes recovers R0 and T0 exactly.
R0 = rotation(math.radians(30.0))
T0 = np.array([100.0, 250.0])


class TestAlignGraph:
    def test_returns_r_t_and_aligned_array(self):
        R, t, Q_aligned = _align_graph(Q_ARR, Q_ARR @ R0 + T0)

        # R is 2x2, t is a length-2 vector, aligned keeps Q's shape
        assert R.shape == (2, 2)
        assert t.shape == (2,)
        assert Q_aligned.shape == Q_ARR.shape

    def test_recovers_known_transform(self):
        G_arr = Q_ARR @ R0 + T0

        R, t, _ = _align_graph(Q_ARR, G_arr)

        # the exact rigid transform is recovered from the paired points
        assert np.allclose(R, R0)
        assert np.allclose(t, T0)

    def test_aligned_lands_on_g(self):
        G_arr = Q_ARR @ R0 + T0

        _, _, Q_aligned = _align_graph(Q_ARR, G_arr)

        # every aligned Q row coincides with its paired G row
        assert np.allclose(Q_aligned, G_arr)

    def test_recovers_reflection(self):
        """A det -1 orthonormal transform (rotation then a y-flip)."""
        Rf = R0 @ np.array([[1.0, 0.0], [0.0, -1.0]])
        G_arr = Q_ARR @ Rf + T0

        R, _, Q_aligned = _align_graph(Q_ARR, G_arr)

        # reflections are allowed, so a mirrored G still aligns exactly
        assert np.allclose(Q_aligned, G_arr)
        # the recovered transform is itself a reflection
        assert np.linalg.det(R) == pytest.approx(-1.0)

    def test_identity_when_q_equals_g(self):
        """Paired points are already coincident."""
        R, t, Q_aligned = _align_graph(Q_ARR, Q_ARR)

        # no rotation and no translation are needed
        assert np.allclose(R, np.eye(2))
        assert np.allclose(t, np.zeros(2))
        # the points are returned unmoved
        assert np.allclose(Q_aligned, Q_ARR)

    def test_does_not_mutate_inputs(self):
        Q_in = Q_ARR.copy()
        G_in = Q_ARR @ R0 + T0
        G_before = G_in.copy()

        _, _, Q_aligned = _align_graph(Q_in, G_in)

        # the inputs are untouched and the result is a fresh array
        assert np.array_equal(Q_in, Q_ARR)
        assert np.array_equal(G_in, G_before)
        assert Q_aligned is not Q_in


class TestDistToSegment:
    # A horizontal segment on the x-axis; easy to reason about feet and distances.
    PA = np.array([0.0, 0.0])
    PB = np.array([4.0, 0.0])

    def test_returns_one_distance_per_row(self):
        coords = np.array([[2.0, 3.0], [1.0, -2.0], [6.0, 0.0]])

        dists = _dist_to_segment(coords, self.PA, self.PB)

        # one distance is returned for each input point
        assert dists.shape == (3,)

    def test_perpendicular_distance_when_foot_is_inside(self):
        """Points whose projection lands between pA and pB."""
        coords = np.array([[2.0, 3.0], [1.0, -2.0], [3.0, 0.5]])

        dists = _dist_to_segment(coords, self.PA, self.PB)

        # distance is the perpendicular offset from the segment line
        assert np.allclose(dists, [3.0, 2.0, 0.5])

    def test_clamps_to_nearest_endpoint_beyond_the_ends(self):
        """Feet fall past pB, past pA, then diagonally past pB."""
        coords = np.array([[6.0, 0.0], [-3.0, 0.0], [7.0, 4.0]])

        dists = _dist_to_segment(coords, self.PA, self.PB)

        # points off the ends measure to the nearer endpoint, not the line
        assert np.allclose(dists, [2.0, 3.0, 5.0])

    def test_zero_on_the_segment(self):
        """The two endpoints and an interior point, all on the segment."""
        coords = np.array([[0.0, 0.0], [4.0, 0.0], [2.0, 0.0]])

        dists = _dist_to_segment(coords, self.PA, self.PB)

        # points lying on the segment are distance zero
        assert np.allclose(dists, [0.0, 0.0, 0.0])

    def test_degenerate_segment_uses_point_distance(self):
        """pA == pB, so the "segment" is a single point."""
        point = np.array([1.0, 1.0])
        coords = np.array([[1.0, 1.0], [4.0, 5.0], [1.0, 4.0]])

        dists = _dist_to_segment(coords, point, point)

        # distance collapses to plain point-to-point Euclidean distance
        assert np.allclose(dists, [0.0, 5.0, 3.0])


def build_graph(coords: dict, edges: list) -> nx.Graph:
    """Build an nx.Graph with x/y node attrs from {id: (x, y)} and an edge list."""
    g = nx.Graph()
    for n, (x, y) in coords.items():
        g.add_node(n, x=float(x), y=float(y))
    g.add_edges_from(edges)
    return g


def coords_of(g: nx.Graph) -> tuple[list, np.ndarray]:
    """Return (node list, coord array) with array row i matching node list[i]."""
    nodes = list(g.nodes)
    arr = np.array([[g.nodes[n]["x"], g.nodes[n]["y"]] for n in nodes])
    return nodes, arr


def find_paths(Q: nx.Graph, G: nx.Graph, match: dict, tol: float):
    """Call _find_paths, deriving the node lists and coord arrays from Q and G."""
    Q_nodes, Q_arr = coords_of(Q)
    G_nodes, G_arr = coords_of(G)
    return _find_paths(Q, G, Q_nodes, G_nodes, Q_arr, G_arr, match, tol)


class TestFindPaths:
    # Q and G share one coordinate frame, so a route lying on a Q segment scores 0.
    # A single Q edge 0->1 spans the x-axis from (0,0) to (4,0); G node ids are
    # offset (10+) to keep them visually distinct from Q ids.
    Q_EDGE = build_graph({0: (0, 0), 1: (4, 0)}, [(0, 1)])
    MATCH = {0: 10, 1: 12}

    def test_returns_score_and_paths_dict(self):
        # straight on-axis route s(0,0) - m(2,0) - t(4,0)
        G = build_graph({10: (0, 0), 11: (2, 0), 12: (4, 0)}, [(10, 11), (11, 12)])

        score, paths = find_paths(self.Q_EDGE, G, self.MATCH, tol=10.0)

        # result unpacks to a float score and a dict of paths
        assert isinstance(score, float)
        assert isinstance(paths, dict)

    def test_paths_keyed_by_query_edges(self):
        """A two-edge query path 0-1-2 routed along a collinear G chain."""
        Q = build_graph({0: (0, 0), 1: (4, 0), 2: (8, 0)}, [(0, 1), (1, 2)])
        G = build_graph(
            {10: (0, 0), 11: (2, 0), 12: (4, 0), 13: (6, 0), 14: (8, 0)},
            [(10, 11), (11, 12), (12, 13), (13, 14)],
        )

        _, paths = find_paths(Q, G, {0: 10, 1: 12, 2: 14}, tol=10.0)

        # one path per Q edge, keyed by the (u, v) tuple
        assert set(paths) == {(0, 1), (1, 2)}

    def test_path_runs_from_matched_source_to_target(self):
        G = build_graph({10: (0, 0), 11: (2, 0), 12: (4, 0)}, [(10, 11), (11, 12)])

        _, paths = find_paths(self.Q_EDGE, G, self.MATCH, tol=10.0)

        # the path starts at match[u] and ends at match[v]
        assert paths[(0, 1)][0] == 10
        assert paths[(0, 1)][-1] == 12

    def test_perfect_alignment_scores_zero(self):
        """Every route node lies exactly on the Q segment."""
        G = build_graph({10: (0, 0), 11: (2, 0), 12: (4, 0)}, [(10, 11), (11, 12)])

        score, paths = find_paths(self.Q_EDGE, G, self.MATCH, tol=10.0)

        # an on-segment route sweeps zero area, along the collinear chain
        assert score == pytest.approx(0.0)
        assert paths[(0, 1)] == [10, 11, 12]

    def test_picks_lowest_area_path(self):
        # straight route (10,11,12) on-axis competes with a detour through 13=(2,3)
        G = build_graph(
            {10: (0, 0), 11: (2, 0), 12: (4, 0), 13: (2, 3)},
            [(10, 11), (11, 12), (10, 13), (13, 12)],
        )

        _, paths = find_paths(self.Q_EDGE, G, self.MATCH, tol=10.0)

        # the zero-area straight route wins over the reachable detour
        assert paths[(0, 1)] == [10, 11, 12]

    def test_detour_area_matches_trapezoid_rule(self):
        """The only route bows out through 13=(2,2), a known swept area."""
        G = build_graph(
            {10: (0, 0), 13: (2, 2), 12: (4, 0)}, [(10, 13), (13, 12)]
        )

        score, paths = find_paths(self.Q_EDGE, G, self.MATCH, tol=10.0)

        # two trapezoids, each 0.5*(0+2)*|edge| with |edge| = 2*sqrt(2)
        assert score == pytest.approx(4.0 * math.sqrt(2.0))
        assert paths[(0, 1)] == [10, 13, 12]

    def test_returns_none_when_no_path(self):
        # source 10 and target 12 sit in disconnected components
        G = build_graph({10: (0, 0), 12: (4, 0), 14: (1, 0)}, [(10, 14)])

        result = find_paths(self.Q_EDGE, G, self.MATCH, tol=10.0)

        # a disconnected match yields no alignment
        assert result is None

    def test_returns_none_when_only_path_exceeds_tol(self):
        # the sole route detours through 13=(2,20), far beyond tol
        G = build_graph({10: (0, 0), 13: (2, 20), 12: (4, 0)}, [(10, 13), (13, 12)])

        result = find_paths(self.Q_EDGE, G, self.MATCH, tol=10.0)

        # every edge of the only route is gated out by the tolerance
        assert result is None

    def test_score_sums_over_edges(self):
        """Each of two query edges routes through its own detour; costs add up."""
        Q = build_graph({0: (0, 0), 1: (4, 0), 2: (8, 0)}, [(0, 1), (1, 2)])
        G = build_graph(
            {10: (0, 0), 13: (2, 2), 12: (4, 0), 15: (6, 1), 14: (8, 0)},
            [(10, 13), (13, 12), (12, 15), (15, 14)],
        )

        score, _ = find_paths(Q, G, {0: 10, 1: 12, 2: 14}, tol=10.0)

        # total is edge (0,1)'s 4*sqrt(2) plus edge (1,2)'s sqrt(5)
        assert score == pytest.approx(4.0 * math.sqrt(2.0) + math.sqrt(5.0))

    def test_endpoint_offset_does_not_block_when_within_tol(self):
        """A route node sitting exactly at tol is allowed; the gate is strict >."""
        G = build_graph({10: (0, 0), 13: (2, 3), 12: (4, 0)}, [(10, 13), (13, 12)])

        result = find_paths(self.Q_EDGE, G, self.MATCH, tol=3.0)

        # distance == tol passes, so the path is found
        assert result is not None
        assert result[1][(0, 1)] == [10, 13, 12]
