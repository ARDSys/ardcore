from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, List, Set

if TYPE_CHECKING:
    from ardcore.knowledge_graph.knowledge_graph import KnowledgeGraph


class NodeMerger(ABC):
    """
    Abstract base class for node merging strategies.
    Implementations will define specific algorithms for identifying
    and merging similar nodes in a knowledge graph.
    """

    @abstractmethod
    def find_merge_candidates(
        self, knowledge_graph: "KnowledgeGraph"
    ) -> List[Set[str]]:
        """
        Find groups of nodes that should be merged together.

        Args:
            knowledge_graph: The KnowledgeGraph instance

        Returns:
            List[Set[str]]: List of sets where each set contains node names to be merged
        """
        pass

    @abstractmethod
    def generate_merged_node_name(
        self, nodes: Set[str], knowledge_graph: "KnowledgeGraph"
    ) -> str:
        """
        Generate a name for the merged node.

        Args:
            nodes: Set of node names to be merged
            knowledge_graph: The KnowledgeGraph instance

        Returns:
            str: The name for the merged node
        """
        pass
