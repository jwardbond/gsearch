import math

import networkx as nx
import pytest

from gsearch.spatial import SpatialGraphIndex

# Node ids are deliberately non-contiguous and out of order so that a mix-up
# between a KDTree row index and a node id cannot pass unnoticed.
#
#   id   coords        distance from node 50
#   50   (0.0, 0.0)     0.0    <- anchor
#   2    (4.0, 0.0)     4.0
#   17   (0.0, 3.0)     3.0
#   8    (3.0, 4.0)     5.0    <- exactly on the r=5 boundary
#   33   (20.0, 20.0)  28.28
#   7    (20.0, 20.0)  28.28   <- coincident with 33
#
# Insertion order (50, 2, 17, 8, 33, 7) differs from distance order
# (50, 17, 2, 8, ...), so index-ordered and distance-ordered results differ.
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


@pytest.fixture
def index(graph: nx.Graph) -> SpatialGraphIndex:
    return SpatialGraphIndex(graph)


def brute_force_within(point: tuple[float, float], radius: float) -> set[int]:
    """Node ids within radius of point, computed without the KDTree."""
    px, py = point
    return {
        node_id
        for node_id, (x, y) in COORDS.items()
        if math.hypot(x - px, y - py) <= radius
    }


class TestSpatialGraphIndex:
    def test_init_copies_graph(self, graph):
        # plan
        index = SpatialGraphIndex(graph)

        # do
        graph.nodes[17]["x"] = 999.0
        graph.add_node(1234, x=1.0, y=1.0)

        # the index holds its own copy, unaffected by later edits
        assert index.graph.nodes[17]["x"] == 0.0
        assert 1234 not in index.graph

    def test_node_ids_align_with_kdtree_rows(self, graph):
        # plan / do
        index = SpatialGraphIndex(graph)

        # node_ids follows graph insertion order
        assert index.node_ids == list(graph.nodes())

        # kdtree row i holds the coordinates of node_ids[i]
        for i, node_id in enumerate(index.node_ids):
            assert tuple(index.kdtree.data[i]) == COORDS[node_id]

    def test_query_radius_distances_align_with_ids(self, index):
        # plan
        point = (1.0, 1.0)

        # do
        dists, node_ids = index.query_radius(point, 25.0)

        # every distance matches its paired node's own coordinates
        assert len(dists) == len(node_ids)
        for dist, node_id in zip(dists, node_ids, strict=True):
            x, y = COORDS[node_id]
            assert dist == pytest.approx(math.hypot(x - point[0], y - point[1]))

    def test_query_radius_sorts_by_distance(self, index):
        # plan
        point = (0.0, 0.0)

        # do
        dists, node_ids = index.query_radius(point, 5.0)

        # results are ordered nearest first, not by node index
        assert node_ids == [50, 17, 2, 8]
        assert dists == pytest.approx([0.0, 3.0, 4.0, 5.0])

    def test_query_radius_returns_exactly_nodes_within_radius(self, index):
        # plan
        point = (1.0, 2.0)
        radius = 6.0

        # do
        _, node_ids = index.query_radius(point, radius)

        # membership matches a brute-force sweep of every node
        assert set(node_ids) == brute_force_within(point, radius)

        # no duplicates in the result
        assert len(node_ids) == len(set(node_ids))

    def test_query_radius_boundary_is_inclusive(self, index):
        # plan
        point = (0.0, 0.0)

        # do
        dists, node_ids = index.query_radius(point, 5.0)

        # node 8 sits at exactly r=5 and is included
        assert 8 in node_ids
        assert dists[node_ids.index(8)] == pytest.approx(5.0)

    def test_query_radius_at_node_coords_includes_that_node(self, index):
        # plan / do
        dists, node_ids = index.query_radius(COORDS[17], 1.0)

        # the node under the query point comes back first, at distance zero
        assert node_ids[0] == 17
        assert dists[0] == pytest.approx(0.0)

    def test_query_radius_returns_coincident_nodes(self, index):
        # plan / do
        dists, node_ids = index.query_radius((20.0, 20.0), 0.5)

        # nodes sharing identical coordinates are both returned
        assert set(node_ids) == {33, 7}
        assert dists == pytest.approx([0.0, 0.0])

    def test_query_radius_no_hits_returns_two_empty_lists(self, index):
        # plan / do
        dists, node_ids = index.query_radius((1000.0, 1000.0), 1.0)

        # an empty result is still an unpackable pair of lists
        assert dists == []
        assert node_ids == []

    def test_make_crop_nodes(self, index):
        # plan / do
        crop = index.make_crop(50, 5.0)

        # the crop holds exactly the nodes within the radius, anchor included
        assert set(crop.nodes()) == brute_force_within(COORDS[50], 5.0)
        assert 50 in crop

    def test_make_crop_edges_are_induced(self, index):
        # plan / do
        crop = index.make_crop(50, 5.0)

        # every edge with both endpoints inside is kept, including 17-8,
        # which touches the anchor only indirectly
        assert set(map(frozenset, crop.edges())) == {
            frozenset((50, 2)),
            frozenset((50, 17)),
            frozenset((17, 8)),
        }

        # the edge 2-33 is dropped because node 33 is outside the radius
        assert not crop.has_edge(2, 33)

    def test_make_crop_preserves_node_attributes(self, index):
        # plan / do
        crop = index.make_crop(50, 5.0)

        # x/y attributes survive the crop unchanged
        for node_id in crop.nodes():
            assert (crop.nodes[node_id]["x"], crop.nodes[node_id]["y"]) == COORDS[
                node_id
            ]

    def test_make_crop_is_unaffected_by_later_source_edits(self, index):
        # plan
        crop = index.make_crop(50, 5.0)

        # do
        index.graph.nodes[17]["x"] = 999.0

        # the crop is a detached copy, not a view onto the indexed graph
        assert crop.nodes[17]["x"] == 0.0

    def test_make_crop_edits_do_not_affect_source(self, index):
        # plan
        crop = index.make_crop(50, 5.0)

        # do
        crop.nodes[17]["x"] = 999.0
        crop.add_node(1234, x=1.0, y=1.0)

        # edits to the crop leave the indexed graph untouched
        assert index.graph.nodes[17]["x"] == 0.0
        assert 1234 not in index.graph

    def test_make_crop_radius_zero(self, index):
        # plan / do
        crop = index.make_crop(50, 0.0)

        # only the anchor itself is within a zero radius
        assert set(crop.nodes()) == {50}
        assert crop.number_of_edges() == 0

    def test_make_crop_radius_zero_keeps_coincident_nodes(self, index):
        # plan / do
        crop = index.make_crop(33, 0.0)

        # a node sharing the anchor's coordinates is at distance zero too
        assert set(crop.nodes()) == {33, 7}
        assert crop.has_edge(33, 7)
