from typing import List

import networkx as nx

from ardcore.knowledge_graph.knowledge_graph import KnowledgeGraph
from ardcore.subgraph.subgraph_generator.base import SubgraphGenerator


class ShortestPathGenerator(SubgraphGenerator):
    """
    Generates a subgraph based on the shortest path between two nodes.
    """

    def generate_path_nodes(
        self, knowledge_graph: KnowledgeGraph, start_node: str, end_node: str
    ) -> List[str]:
        """
        Generate a list of nodes that form the shortest path between start_node and end_node.

        Args:
            knowledge_graph: The KnowledgeGraph instance
            start_node: The starting node for the path
            end_node: The ending node for the path

        Returns:
            List[str]: List of node names that form the shortest path

        Raises:
            nx.NetworkXNoPath: If no path exists between start_node and end_node
        """
        # Validate nodes
        self.validate_nodes(knowledge_graph, start_node, end_node)

        # Find the shortest path between start and end nodes
        try:
            path = knowledge_graph.graph.shortest_path(
                source=start_node,
                target=end_node,
                directed=False,
                scientific_domain=knowledge_graph.scientific_domain,
            )
            return path
        except nx.NetworkXNoPath:
            raise nx.NetworkXNoPath(
                f"No path exists between '{start_node}' and '{end_node}'"
            )
