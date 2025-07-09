import random
from heapq import heappop, heappush
from typing import Dict, List, Optional

import networkx as nx
import numpy as np

from ardcore.knowledge_graph import KnowledgeGraph
from ardcore.subgraph.subgraph_generator import SubgraphGenerator
from ardcore.utils.embedder import Embedder


class RandomizedEmbeddingPathGenerator(SubgraphGenerator):
    """
    Generates a subgraph based on embeddings with added randomization and waypoints.

    This generator uses node embeddings to guide the search but introduces randomness
    and intermediate waypoints to create more diverse and potentially insightful paths
    between nodes.
    """

    def __init__(
        self,
        node_embeddings: Optional[Dict[str, np.ndarray]] = None,
        embedder: Optional[Embedder] = None,
        top_k: int = 3,
        randomness_factor: float = 0.5,
        num_random_waypoints: int = 3,
        seed: Optional[int] = None,
    ):
        """
        Initialize the RandomizedEmbeddingPathGenerator.

        Args:
            node_embeddings: Dictionary mapping node names to their embeddings (optional if embedder is provided)
            embedder: Embedder instance to use for generating embeddings (optional if node_embeddings is provided)
            top_k: Number of top neighbors to consider at each step
            randomness_factor: Factor between 0 and 1 controlling the balance between
                               heuristic-based pathfinding (0) and randomness (1)
            num_random_waypoints: Number of random waypoints to introduce into the path
            seed: Random seed for reproducibility
        """
        if node_embeddings is None and embedder is None:
            raise ValueError("Either node_embeddings or embedder must be provided")

        self.node_embeddings = node_embeddings or {}
        self.embedder = embedder
        self.top_k = top_k
        self.randomness_factor = max(
            0.0, min(1.0, randomness_factor)
        )  # Clamp between 0 and 1
        self.num_random_waypoints = num_random_waypoints
        self.seed = seed
        if seed is not None:
            random.seed(seed)

    def _heuristic(self, current: str, target: str) -> float:
        """
        Estimate distance from current to target node using embeddings.

        Args:
            current: Current node name
            target: Target node name

        Returns:
            float: Estimated distance
        """
        if self.embedder:
            return self.embedder.calculate_distance(current, target, metric="euclidean")
        else:
            # Use euclidean distance if no embedder is provided
            return np.linalg.norm(
                self.node_embeddings[current] - self.node_embeddings[target]
            )

    def _dijkstra_with_randomness(
        self, knowledge_graph: KnowledgeGraph, source: str, target: str
    ) -> Optional[List[str]]:
        """
        Find a path using Dijkstra's algorithm with added randomness.

        Args:
            knowledge_graph: The knowledge graph to search
            source: Source node
            target: Target node

        Returns:
            Optional[List[str]]: Path from source to target, or None if no path exists
        """
        queue = [(0, source, [])]
        visited = set()

        while queue:
            (cost, node, path) = heappop(queue)

            if node not in visited:
                visited.add(node)
                path = path + [node]

                if node == target:
                    return path

                neighbors = list(knowledge_graph.get_node_neighbors(node))
                random.shuffle(neighbors)

                for neighbor in neighbors:
                    if neighbor not in visited:
                        # Get edge weight if available, default to 1
                        # edge_weight = knowledge_graph.get_edge_attrs(node, neighbor).get("weight", 1)
                        edge_weight = 1
                        new_cost = cost + edge_weight

                        # Add randomness to the priority
                        priority = new_cost + self.randomness_factor * random.random()

                        heappush(queue, (priority, neighbor, path))

        return None

    def _add_random_waypoints(
        self, knowledge_graph: KnowledgeGraph, path: List[str], target: str
    ) -> List[str]:
        """
        Add random waypoints to a path to create a more diverse route.

        Args:
            knowledge_graph: The knowledge graph to search
            path: Initial path (at least containing the source node)
            target: Target node

        Returns:
            List[str]: Path with random waypoints
        """
        if not path:
            return []

        # Collect all neighbors of nodes in the path
        all_neighbors = []
        for node in path:
            all_neighbors.extend(
                [
                    neighbor
                    for neighbor in knowledge_graph.get_node_neighbors(node)
                    if neighbor not in path
                ]
            )

        # Shuffle and select waypoints
        random.shuffle(all_neighbors)
        waypoints = all_neighbors[: min(self.num_random_waypoints, len(all_neighbors))]

        # Start with the source node
        new_path = path[:1]

        # Add paths to each waypoint
        for waypoint in waypoints:
            try:
                # Find shortest path to the waypoint
                waypoint_path = knowledge_graph.graph.shortest_path(
                    source=new_path[-1],
                    target=waypoint,
                    directed=False,
                    scientific_domain=knowledge_graph.scientific_domain,
                )
                # Add the path (excluding the first node to avoid duplication)
                new_path.extend(waypoint_path[1:])
            except nx.NetworkXNoPath:
                # If no path to waypoint, skip it
                continue

        # Add the final leg to the target
        try:
            final_leg = knowledge_graph.graph.shortest_path(
                source=new_path[-1],
                target=target,
                directed=False,
                scientific_domain=knowledge_graph.scientific_domain,
            )
            # Add the final leg (excluding the first node to avoid duplication)
            new_path.extend(final_leg[1:])
        except nx.NetworkXNoPath:
            # If no path to target from last waypoint, try direct from source
            if new_path[0] != target:
                try:
                    direct_path = knowledge_graph.graph.shortest_path(
                        source=new_path[0],
                        target=target,
                        directed=False,
                        scientific_domain=knowledge_graph.scientific_domain,
                    )
                    new_path = direct_path
                except nx.NetworkXNoPath:
                    # If still no path, return what we have
                    pass

        return new_path

    def generate_path_nodes(
        self, knowledge_graph: KnowledgeGraph, start_node: str, end_node: str
    ) -> List[str]:
        """
        Generate a list of nodes that form a path between start_node and end_node
        using embeddings with randomization and waypoints.

        Args:
            knowledge_graph: The KnowledgeGraph instance
            start_node: The starting node for the path
            end_node: The ending node for the path

        Returns:
            List[str]: List of node names that form the path

        Raises:
            ValueError: If start_node or end_node are not in the graph
            nx.NetworkXNoPath: If no path exists between start_node and end_node
        """
        # Validate nodes
        self.validate_nodes(knowledge_graph, start_node, end_node)

        # If we have an embedder but no pre-calculated embeddings, calculate them now
        if self.embedder and self.embedder.embeddings_len == 0:
            words = list(knowledge_graph.get_nodes())
            self.embedder.embed(words)

        # Find initial path based on randomness factor
        if self.randomness_factor == 0:
            # Use standard Dijkstra's algorithm for the shortest path
            try:
                path = knowledge_graph.graph.shortest_path(
                    source=start_node,
                    target=end_node,
                    directed=False,
                    scientific_domain=knowledge_graph.scientific_domain,
                )
            except nx.NetworkXNoPath:
                raise nx.NetworkXNoPath(
                    f"No path found between '{start_node}' and '{end_node}'"
                )
        else:
            # Use Dijkstra with randomness
            path = self._dijkstra_with_randomness(knowledge_graph, start_node, end_node)
            if path is None:
                raise nx.NetworkXNoPath(
                    f"No path found between '{start_node}' and '{end_node}'"
                )

        # Add random waypoints if requested
        if self.num_random_waypoints > 0:
            path = self._add_random_waypoints(knowledge_graph, path, end_node)

            # Ensure the path starts with start_node and ends with end_node
            if path and path[0] != start_node:
                path.insert(0, start_node)
            if path and path[-1] != end_node:
                path.append(end_node)

        # Verify the path is valid
        if not path or path[0] != start_node or path[-1] != end_node:
            raise nx.NetworkXNoPath(
                f"Failed to generate a valid path between '{start_node}' and '{end_node}'"
            )

        return path
