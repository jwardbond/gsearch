import math

import networkx as nx
import pytest

from gsearch.core import _distance_from_node

# Node ids are deliberately non-contiguous and out of order so that a mix-up
# between an insertion index and a node id cannot pass unnoticed.
#
#   id   coords        distance from node 50
#   50   (0.0, 0.0)     0.0    <- anchor
#   2    (4.0, 0.0)     4.0
#   17   (0.0, 3.0)     3.0
#   8    (3.0, 4.0)     5.0
#   33   (20.0, 20.0)  28.28
#   7    (20.0, 20.0)  28.28   <- coincident with 33
COORDS = {
    50: (0.0, 0.0),
    2: (4.0, 0.0),
    17: (0.0, 3.0),
    8: (3.0, 4.0),
    33: (20.0, 20.0),
    7: (20.0, 20.0),
}
EDGES = [(50, 2), (50, 17), (17, 8), (2, 33), (33, 7)]


@pytest.fixture
def graph() -> nx.Graph:
    g = nx.Graph()
    for node_id, (x, y) in COORDS.items():
        g.add_node(node_id, x=x, y=y)
    g.add_edges_from(EDGES)
    return g


def brute_force_distances(anchor_id: int) -> dict[int, float]:
    """Euclidean distances from anchor to every other node, computed directly."""
    ax, ay = COORDS[anchor_id]
    return {
        node_id: math.hypot(x - ax, y - ay)
        for node_id, (x, y) in COORDS.items()
        if node_id != anchor_id
    }


class TestDistanceFromNode:
    def test_excludes_anchor(self, graph):
        result = _distance_from_node(50, graph)

        # the anchor node is not among its own distances
        assert 50 not in [node_id for node_id, _ in result]

    def test_covers_all_other_nodes(self, graph):
        result = _distance_from_node(50, graph)

        # every other node appears exactly once
        node_ids = [node_id for node_id, _ in result]
        assert set(node_ids) == set(COORDS) - {50}
        assert len(node_ids) == len(set(node_ids))

    def test_distances_match_coordinates(self, graph):
        result = _distance_from_node(50, graph)

        # each distance matches its paired node, checked against a brute-force sweep
        expected = brute_force_distances(50)
        for node_id, dist in result:
            assert dist == pytest.approx(expected[node_id])

    def test_sorted_nearest_first(self, graph):
        result = _distance_from_node(50, graph)

        # distances come back in non-decreasing order
        dists = [dist for _, dist in result]
        assert dists == sorted(dists)

    def test_known_distances(self, graph):
        result = _distance_from_node(50, graph)

        # exact hand-computed order and values from anchor 50 (33/7 tie, stable)
        assert [node_id for node_id, _ in result] == [17, 2, 8, 33, 7]
        assert [dist for _, dist in result] == pytest.approx(
            [3.0, 4.0, 5.0, math.hypot(20, 20), math.hypot(20, 20)]
        )

    def test_coincident_node_is_zero(self, graph):
        result = _distance_from_node(33, graph)

        # node 7 shares the anchor's coords, so its distance is zero (not excluded)
        assert dict(result)[7] == pytest.approx(0.0)

    def test_single_node_graph_returns_empty(self):
        g = nx.Graph()
        g.add_node(50, x=0.0, y=0.0)

        result = _distance_from_node(50, g)

        # a lone anchor has no other nodes to measure
        assert result == []
