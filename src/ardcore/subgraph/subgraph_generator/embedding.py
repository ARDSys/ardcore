import random
from typing import Dict, List, Optional

import networkx as nx
import numpy as np

from ardcore.knowledge_graph import KnowledgeGraph
from ardcore.subgraph.subgraph_generator.base import SubgraphGenerator
from ardcore.utils.embedder import Embedder


class EmbeddingPathGenerator(SubgraphGenerator):
    """
    Generates a subgraph based on embeddings to guide a heuristic search between two nodes.
    Uses node embeddings to estimate distances and guide the search towards the target node.
    """

    def __init__(
        self,
        node_embeddings: Optional[Dict[str, np.ndarray]] = None,
        embedder: Optional[Embedder] = None,
        top_k: int = 3,
        seed: Optional[int] = None,
    ):
        """
        Initialize the EmbeddingPathGenerator.

        Args:
            node_embeddings: Dictionary mapping node names to their embeddings (optional if embedder is provided)
            embedder: Embedder instance to use for generating embeddings (optional if node_embeddings is provided)
            top_k: Number of top neighbors to consider at each step
            seed: Random seed for reproducibility
        """
        if node_embeddings is None and embedder is None:
            raise ValueError("Either node_embeddings or embedder must be provided")

        self.node_embeddings = node_embeddings or {}
        self.embedder = embedder
        self.top_k = top_k
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
            return self.embedder.calculate_distance(current, target)
        else:
            # Use euclidean distance if no embedder is provided
            return np.linalg.norm(
                self.node_embeddings[current] - self.node_embeddings[target]
            )

    def generate_path_nodes(
        self, knowledge_graph: KnowledgeGraph, start_node: str, end_node: str
    ) -> List[str]:
        """
        Generate a list of nodes that form a path between start_node and end_node using embeddings.

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
        if self.embedder and not self.node_embeddings:
            words = list(knowledge_graph.get_nodes())
            self.node_embeddings = self.embedder.get_embeddings(words)

        def sample_path(current: str, visited: set) -> Optional[List[str]]:
            """
            Sample a path from current node to end_node using heuristic search.

            Args:
                current: Current node
                visited: Set of visited nodes

            Returns:
                Optional[List[str]]: Path from current to end_node, or None if no path exists
            """
            path = [current]

            while current != end_node:
                # Get neighbors and their heuristic values
                neighbors = [
                    (neighbor, self._heuristic(neighbor, end_node))
                    for neighbor in knowledge_graph.get_node_neighbors(current)
                    if neighbor not in visited
                ]

                if not neighbors:
                    # Dead end reached, backtrack if possible
                    if len(path) > 1:
                        visited.add(
                            path.pop()
                        )  # Mark the dead-end node as visited and remove it
                        current = path[-1]  # Backtrack to the previous node
                        continue
                    else:
                        # No path found
                        return None
                else:
                    # Sort neighbors by heuristic value (lower is better)
                    neighbors.sort(key=lambda x: x[1])

                    # Select from top_k neighbors
                    top_neighbors = (
                        neighbors[: self.top_k]
                        if len(neighbors) > self.top_k
                        else neighbors
                    )
                    next_node = random.choice(top_neighbors)[0]

                    path.append(next_node)
                    visited.add(next_node)
                    current = next_node

                    # Prevent infinite loops
                    if len(path) > 2 * knowledge_graph.number_of_nodes():
                        raise nx.NetworkXNoPath(
                            f"No path found between '{start_node}' and '{end_node}'"
                        )

            return path

        # Start the search
        visited = set([start_node])
        path = sample_path(start_node, visited)

        if path is None:
            raise nx.NetworkXNoPath(
                f"No path found between '{start_node}' and '{end_node}'"
            )

        return path
