import math

import numpy as np
import pytest

from gsearch.align_edges import _align_graph, _dist_to_segment

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
