import math

import numpy as np
import pytest

from gsearch.align_nodes import create_annulus_mask

# A shared reference point plus points chosen for their DISTANCE from pt, since
# the mask depends only on distance-to-pt, never on position.
#
#   arr1 (rows)                 distance from pt (5, 5)
#   (5.0, 8.0)                   3.0
#   (9.0, 5.0)                   4.0
#
#   arr2 (cols)                 distance from pt (5, 5)
#   (5.0, 2.0)                   3.0    <- same dist as arr1[0], different angle
#   (5.0, 9.0)                   4.0    <- same dist as arr1[1]
#   (5.0, 7.0)                   2.0
#   (30.0, 5.0)                 25.0    <- far from every arr1 distance
PT = np.array([5.0, 5.0])
ARR1 = np.array([[5.0, 8.0], [9.0, 5.0]])
ARR2 = np.array([[5.0, 2.0], [5.0, 9.0], [5.0, 7.0], [30.0, 5.0]])


def brute_force_mask(
    arr1: np.ndarray, arr2: np.ndarray, pt: np.ndarray, tol: float
) -> np.ndarray:
    """Annulus mask computed with plain Python loops, no broadcasting."""
    mask = np.zeros((len(arr1), len(arr2)), dtype=bool)
    for i, (ax, ay) in enumerate(arr1):
        d1 = math.hypot(ax - pt[0], ay - pt[1])
        for j, (bx, by) in enumerate(arr2):
            d2 = math.hypot(bx - pt[0], by - pt[1])
            mask[i, j] = abs(d1 - d2) <= tol
    return mask


class TestCreateAnnulusMask:
    def test_output_shape(self):
        # plan / do
        mask = create_annulus_mask(ARR1, ARR2, PT, tol=0.5)

        # one row per arr1 point, one column per arr2 point
        assert mask.shape == (len(ARR1), len(ARR2))

    def test_matches_brute_force(self):
        # plan
        tol = 0.5

        # do
        mask = create_annulus_mask(ARR1, ARR2, PT, tol=tol)

        # every entry agrees with an independent loop computation
        assert np.array_equal(mask, brute_force_mask(ARR1, ARR2, PT, tol))

    def test_rotation_invariant_around_pt(self):
        # plan / do
        mask = create_annulus_mask(ARR1, ARR2, PT, tol=0.0)

        # arr2[0] sits at the same distance as arr1[0] but a different angle,
        # yet still matches: the annulus is about distance, not position
        assert mask[0, 0]
        # likewise arr2[1] matches arr1[1] purely by shared distance
        assert mask[1, 1]

    def test_boundary_is_inclusive(self):
        # plan: arr1[0] is at distance 3, arr2[2] at distance 2 -> diff 1.0
        tol = 1.0

        # do
        mask = create_annulus_mask(ARR1, ARR2, PT, tol=tol)

        # a pair whose distance difference equals tol exactly is True
        assert mask[0, 2]

    def test_tol_zero_requires_exact_distance(self):
        # plan / do
        mask = create_annulus_mask(ARR1, ARR2, PT, tol=0.0)

        # only equal-distance pairs survive a zero tolerance
        assert np.array_equal(mask, brute_force_mask(ARR1, ARR2, PT, tol=0.0))
        # the distance-2 and distance-25 columns match neither arr1 point
        assert not mask[:, 2].any()
        assert not mask[:, 3].any()

    def test_point_outside_all_annuli_is_all_false(self):
        # plan / do
        mask = create_annulus_mask(ARR1, ARR2, PT, tol=0.5)

        # arr2[3] is far from every arr1 distance, so its column is all False
        assert not mask[:, 3].any()

    def test_empty_arr1_raises(self):
        # plan
        empty = np.empty((0, 2))

        # do / test: an empty arr1 has no annuli to test against
        with pytest.raises(ValueError, match="arr1"):
            create_annulus_mask(empty, ARR2, PT, tol=0.5)

    def test_empty_arr2_raises(self):
        # plan
        empty = np.empty((0, 2))

        # do / test: an empty arr2 has no points to place in the annuli
        with pytest.raises(ValueError, match="arr2"):
            create_annulus_mask(ARR1, empty, PT, tol=0.5)
