import random
from typing import List

import networkx as nx

from ard.knowledge_graph import KnowledgeGraph
from ard.subgraph.subgraph_generator import (
    SingleNodeSubgraphGenerator,
    SubgraphGenerator,
)


class RandomWalkGenerator(SubgraphGenerator):
    """
    Generates a subgraph based on a random walk from the start node.
    If the end node is found during the walk, the walk terminates.
    Otherwise, it continues until max_steps is reached.
    """

    def __init__(self, max_steps: int = 10, seed: int = None):
        """
        Initialize the RandomWalkGenerator.

        Args:
            max_steps: Maximum number of steps for the random walk
            seed: Random seed for reproducibility
        """
        self.max_steps = max_steps
        self.seed = seed
        if seed is not None:
            random.seed(seed)

    def generate_path_nodes(
        self, knowledge_graph: KnowledgeGraph, start_node: str, end_node: str
    ) -> List[str]:
        """
        Generate a list of nodes from a random walk starting at start_node.
        If end_node is encountered during the walk, the walk terminates at that point.

        Args:
            knowledge_graph: The KnowledgeGraph instance
            start_node: The starting node for the path
            end_node: The target ending node for the path

        Returns:
            List[str]: List of node names that form the path

        Raises:
            nx.NetworkXNoPath: If no path to the end node was found within max_steps
        """
        # Validate nodes
        self.validate_nodes(knowledge_graph, start_node, end_node)

        # Start the random walk
        path = [start_node]
        current_node = start_node

        for _ in range(self.max_steps):
            # Get neighbors of the current node (both predecessors and successors)
            neighbors = knowledge_graph.get_node_neighbors(current_node)

            # If there are no neighbors, we're stuck
            if not neighbors:
                break

            # Select a random neighbor
            next_node = random.choice(neighbors)
            path.append(next_node)

            # If we've reached the end node, we're done
            if next_node == end_node:
                return path

            # Otherwise, continue the walk
            current_node = next_node

        # If we've gone through all steps and haven't found the end node,
        # check if we can reach it directly from the last node we visited
        all_neighbors = knowledge_graph.get_node_neighbors(current_node)

        if end_node in all_neighbors:
            path.append(end_node)
            return path

        # If we didn't reach the end node, raise an exception
        if end_node not in path:
            raise nx.NetworkXNoPath(
                f"Random walk did not reach '{end_node}' from '{start_node}' within {self.max_steps} steps"
            )

        # Return the path we found (if end_node is in the path)
        return path


class SingleNodeRandomWalkGenerator(SingleNodeSubgraphGenerator):
    """
    Generates a subgraph based on a random walk from a single start node.
    The walk continues until max_steps is reached.
    """

    def __init__(self, max_steps: int = 10, seed: int = None):
        """
        Initialize the SingleNodeRandomWalkGenerator.

        Args:
            max_steps: Maximum number of steps for the random walk
            seed: Random seed for reproducibility
        """
        self.max_steps = max_steps
        self.seed = seed
        if seed is not None:
            random.seed(seed)

    def generate_path_nodes(
        self, knowledge_graph: KnowledgeGraph, start_node: str
    ) -> List[str]:
        """
        Generate a list of nodes from a random walk starting at start_node.
        The walk continues until max_steps is reached or there are no more neighbors.

        Args:
            knowledge_graph: The KnowledgeGraph instance
            start_node: The starting node for the path

        Returns:
            List[str]: List of node names that form the path
        """
        # Validate node
        self.validate_node(knowledge_graph, start_node)

        # Start the random walk
        path = [start_node]
        current_node = start_node

        for _ in range(self.max_steps):
            # Get neighbors of the current node (both predecessors and successors)
            neighbors = knowledge_graph.get_node_neighbors(current_node)

            # If there are no neighbors, we're stuck
            if not neighbors:
                break

            # Select a random neighbor
            next_node = random.choice(neighbors)
            path.append(next_node)

            # Continue the walk
            current_node = next_node

        return path
