import networkx as nx
import numpy as np
from scipy.spatial import KDTree


class SpatialGraphIndex:
    def __init__(self, graph: nx.Graph):
        self.graph = graph.copy()
        self.kdtree, self.node_ids = self._build_index(graph)

    def _build_index(self, graph: nx.Graph) -> tuple[KDTree, list[int]]:
        """Build a KDTree index for the nodes in the graph."""
        node_ids = list(graph.nodes())
        coords = [(graph.nodes[n]["x"], graph.nodes[n]["y"]) for n in node_ids]
        return KDTree(coords), node_ids

    def query_radius(
        self, point: tuple[float, float], radius: float
    ) -> tuple[list[float], list[int]]:
        """Query the KDTree for all nodes within a given radius of a point.

        Returns:
            A tuple of
            - distances: list of distances from the point to each node within the radius
            - node_ids: list of node IDs corresponding to the distances
        """
        idxs = self.kdtree.query_ball_point(point, r=radius)

        if not idxs:
            return [], []

        p = np.asarray(point)

        dists = np.linalg.norm(self.kdtree.data[idxs] - p, axis=1)

        # query_ball_poi`nt sorts by index, so re-sort both lists by distance
        order = np.argsort(dists)
        dists = dists[order]
        idxs = np.asarray(idxs)[order]

        return dists.tolist(), [self.node_ids[i] for i in idxs]

    def make_crop(self, node_id: int, radius: float):
        """Crop the graph to include only nodes within a given radius of a specified node."""

        node_data = self.graph.nodes[node_id]
        point = (node_data["x"], node_data["y"])

        _, neighbourhood = self.query_radius(point, radius)

        # Create a subgraph with the nearby nodes
        cropped_graph = self.graph.subgraph(neighbourhood).copy()

        # Keep only the component that contains the anchor node.
        component = nx.node_connected_component(cropped_graph, node_id)
        cropped_graph = cropped_graph.subgraph(component).copy()

        return cropped_graph
