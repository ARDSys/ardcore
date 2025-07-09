from abc import ABC, abstractmethod
from typing import List

from ardcore.knowledge_graph.knowledge_graph import KnowledgeGraph


class SubgraphGenerator(ABC):
    """
    Abstract base class for subgraph generation strategies.
    Implementations will define specific algorithms for generating
    subgraphs between nodes in a knowledge graph.
    """

    @abstractmethod
    def generate_path_nodes(
        self, knowledge_graph: KnowledgeGraph, start_node: str, end_node: str
    ) -> List[str]:
        """
        Generate a list of nodes that form a path between start_node and end_node.

        Args:
            knowledge_graph: The KnowledgeGraph instance
            start_node: The starting node for the path
            end_node: The ending node for the path

        Returns:
            List[str]: List of node names that form the path
        """
        pass

    def validate_nodes(
        self, knowledge_graph: KnowledgeGraph, start_node: str, end_node: str
    ) -> None:
        """
        Validate that the start and end nodes exist in the graph.

        Args:
            knowledge_graph: The KnowledgeGraph instance
            start_node: The starting node for the path
            end_node: The ending node for the path

        Raises:
            ValueError: If start_node or end_node are not in the graph
        """
        if not knowledge_graph.has_node(start_node):
            raise ValueError(f"Start node '{start_node}' not found in the graph")
        if not knowledge_graph.has_node(end_node):
            raise ValueError(f"End node '{end_node}' not found in the graph")


class SingleNodeSubgraphGenerator(ABC):
    """
    Abstract base class for single-node subgraph generation strategies.
    Implementations will define specific algorithms for generating
    subgraphs starting from a single node in a knowledge graph.
    """

    @abstractmethod
    def generate_path_nodes(
        self, knowledge_graph: KnowledgeGraph, start_node: str
    ) -> List[str]:
        """
        Generate a list of nodes that form a path starting from start_node.

        Args:
            knowledge_graph: The KnowledgeGraph instance
            start_node: The starting node for the path

        Returns:
            List[str]: List of node names that form the path
        """
        pass

    def validate_node(self, knowledge_graph: KnowledgeGraph, start_node: str) -> None:
        """
        Validate that the start node exists in the graph.

        Args:
            knowledge_graph: The KnowledgeGraph instance
            start_node: The starting node for the path

        Raises:
            ValueError: If start_node is not in the graph
        """
        if not knowledge_graph.has_node(start_node):
            raise ValueError(f"Start node '{start_node}' not found in the graph")
