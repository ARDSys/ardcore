from abc import ABC, abstractmethod
from typing import Any, Dict, List, Set, Tuple


class GraphBackend(ABC):
    """
    Abstract base class for graph backends.
    This defines the interface that all graph backends must implement.
    """

    @abstractmethod
    def add_node(self, node: str, **attrs) -> None:
        """Add a node with optional attributes."""
        pass

    @abstractmethod
    def has_node(self, node: str) -> bool:
        """Check if a node exists."""
        pass

    @abstractmethod
    def add_edge(self, source: str, target: str, **attrs) -> None:
        """Add an edge with optional attributes."""
        pass

    @abstractmethod
    def has_edge(self, source: str, target: str) -> bool:
        """Check if an edge exists."""
        pass

    @abstractmethod
    def get_node_attrs(self, node: str) -> Dict[str, Any]:
        """Get all attributes of a node."""
        pass

    @abstractmethod
    def get_edge_attrs(self, source: str, target: str) -> Dict[str, Any]:
        """Get all attributes of an edge."""
        pass

    @abstractmethod
    def get_nodes(self) -> Set[str]:
        """Get all nodes in the graph."""
        pass

    @abstractmethod
    def get_edges(self) -> List[Tuple[str, str, Dict[str, Any]]]:
        """Get all edges in the graph with their attributes."""
        pass

    @abstractmethod
    def get_successors(self, node: str) -> List[str]:
        """Get all successor nodes of a node."""
        pass

    @abstractmethod
    def get_predecessors(self, node: str) -> List[str]:
        """Get all predecessor nodes of a node."""
        pass

    @abstractmethod
    def get_out_edges(self, node: str) -> List[Tuple[str, str, Dict[str, Any]]]:
        """Get all outgoing edges of a node with their attributes."""
        pass

    @abstractmethod
    def get_in_edges(self, node: str) -> List[Tuple[str, str, Dict[str, Any]]]:
        """Get all incoming edges of a node with their attributes."""
        pass

    @abstractmethod
    def remove_node(self, node: str) -> None:
        """Remove a node and all its edges."""
        pass

    @abstractmethod
    def number_of_edges(self) -> int:
        """Get the total number of edges in the graph."""
        pass

    @abstractmethod
    def shortest_path(
        self, source: str, target: str, directed: bool = True
    ) -> List[str]:
        """Get the shortest path between two nodes."""
        pass

    @abstractmethod
    def __len__(self) -> int:
        """Get the total number of nodes in the graph."""
        pass

    @abstractmethod
    def to_serializable(self) -> Dict[str, Any]:
        """
        Convert the graph to a serializable dictionary format.

        This method should create a dictionary that can be serialized to JSON,
        MessagePack, or other formats, and later reconstructed using the
        from_serializable class method.

        Returns:
            Dict[str, Any]: A serializable representation of the graph
        """
        pass
