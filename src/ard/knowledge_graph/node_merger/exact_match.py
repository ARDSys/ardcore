from typing import List, Set

from ard.knowledge_graph.node_merger.base import NodeMerger


class ExactMatchNodeMerger(NodeMerger):
    """
    Merges nodes with exactly matching names (case-insensitive).

    This merger identifies nodes that are identical when converted to lowercase
    and groups them for merging.
    """

    def find_merge_candidates(self, knowledge_graph) -> List[Set[str]]:
        """
        Find nodes with identical names (case-insensitive).

        Args:
            knowledge_graph: The KnowledgeGraph instance

        Returns:
            List[Set[str]]: List of sets where each set contains node names to be merged
        """
        # Group nodes by lowercase name
        nodes_by_lowercase = {}
        for node in knowledge_graph.get_nodes():
            lower = node.lower()
            if lower not in nodes_by_lowercase:
                nodes_by_lowercase[lower] = set()
            nodes_by_lowercase[lower].add(node)

        # Return only groups with more than one node
        return [group for group in nodes_by_lowercase.values() if len(group) > 1]

    def generate_merged_node_name(self, nodes: Set[str], knowledge_graph) -> str:
        """
        Generate a name for the merged node, using the most frequent capitalization.

        Args:
            nodes: Set of node names to be merged
            knowledge_graph: The KnowledgeGraph instance

        Returns:
            str: The name for the merged node
        """
        # Pick the most frequently used capitalization
        node_counts = {}
        for node in nodes:
            # Count occurrences in sources
            node_attrs = knowledge_graph.get_node_attrs(node)
            count = len(node_attrs.get("sources", []))
            node_counts[node] = count

        # Return the node with highest count, or the first one if counts are equal
        return max(node_counts.items(), key=lambda x: x[1])[0]
