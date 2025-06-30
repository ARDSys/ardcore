from typing import TYPE_CHECKING, Any, List, Set, Tuple

from ard.knowledge_graph.node_merger.base import NodeMerger

if TYPE_CHECKING:
    from ard.knowledge_graph.knowledge_graph import KnowledgeGraph


class LLMBasedNodeMerger(NodeMerger):
    """
    Merges nodes based on LLM judgment.

    This merger uses an LLM to decide whether nodes should be merged.
    It requires an LLM provider that can be called via a function.
    """

    def __init__(
        self,
        llm_provider: Any,
        similarity_threshold: float = 0.5,
        max_comparisons: int = 1000,
    ):
        """
        Initialize the LLM-based node merger.

        Args:
            llm_provider: Object with a method to call the LLM
            similarity_threshold: Minimum similarity score to consider merging (0-1)
            max_comparisons: Maximum number of node pairs to compare
        """
        self.llm_provider = llm_provider
        self.similarity_threshold = similarity_threshold
        self.max_comparisons = max_comparisons

    def find_merge_candidates(
        self, knowledge_graph: "KnowledgeGraph"
    ) -> List[Set[str]]:
        """
        Find groups of nodes that an LLM judges to be similar enough to merge.

        Args:
            knowledge_graph: The KnowledgeGraph instance

        Returns:
            List[Set[str]]: List of sets where each set contains node names to be merged
        """
        nodes = list(knowledge_graph.get_nodes())
        merge_pairs = []

        # Limit the number of comparisons for practical reasons
        import itertools

        comparisons = list(itertools.combinations(nodes, 2))[: self.max_comparisons]

        for node1, node2 in comparisons:
            # Query the LLM to determine if nodes should be merged
            prompt = self._generate_merge_prompt(node1, node2, knowledge_graph)
            result = self.llm_provider.query(prompt)
            score = self._parse_llm_response(result)

            if score >= self.similarity_threshold:
                merge_pairs.append((node1, node2))

        # Convert pairs to groups (handling transitive relationships)
        return self._convert_pairs_to_groups(merge_pairs, nodes)

    def generate_merged_node_name(
        self, nodes: Set[str], knowledge_graph: "KnowledgeGraph"
    ) -> str:
        """
        Generate a name for the merged node using the LLM.

        Args:
            nodes: Set of node names to be merged
            knowledge_graph: The KnowledgeGraph instance

        Returns:
            str: The name for the merged node
        """
        # Query the LLM to generate a merged name
        prompt = self._generate_name_prompt(nodes, knowledge_graph)
        result = self.llm_provider.query(prompt)

        # Parse the result to get the suggested name
        merged_name = result.strip()

        # Fall back to first node if LLM doesn't provide a usable name
        if not merged_name:
            return next(iter(nodes))

        return merged_name

    def _generate_merge_prompt(
        self, node1: str, node2: str, knowledge_graph: "KnowledgeGraph"
    ) -> str:
        """
        Generate a prompt for the LLM to decide if two nodes should be merged.

        Args:
            node1: First node name
            node2: Second node name
            knowledge_graph: The KnowledgeGraph instance

        Returns:
            str: Prompt for the LLM
        """
        # Get node context (neighbors and edges)
        node1_neighbors = knowledge_graph.get_node_neighbors_relations(node1)
        node2_neighbors = knowledge_graph.get_node_neighbors_relations(node2)

        # Format the prompt
        prompt = f"""
        I have two nodes in a knowledge graph that may represent the same concept:
        
        Node 1: "{node1}"
        Relationships of Node 1:
        {self._format_relationships(node1_neighbors)}
        
        Node 2: "{node2}"
        Relationships of Node 2:
        {self._format_relationships(node2_neighbors)}
        
        On a scale of 0 to 1, where 0 means completely different concepts and 1 means exactly the same concept,
        how similar are these nodes? Provide only a numeric score.
        """

        return prompt

    def _generate_name_prompt(
        self, nodes: Set[str], knowledge_graph: "KnowledgeGraph"
    ) -> str:
        """
        Generate a prompt for the LLM to suggest a name for the merged node.

        Args:
            nodes: Set of node names to be merged
            knowledge_graph: The KnowledgeGraph instance

        Returns:
            str: Prompt for the LLM
        """
        # Format the prompt
        nodes_list = ", ".join([f'"{node}"' for node in nodes])
        prompt = f"""
        I have determined that the following nodes in a knowledge graph represent the same concept: {nodes_list}
        
        Please suggest a single, clear name that best represents this concept. Provide only the name without explanations.
        """

        return prompt

    def _format_relationships(self, relationships: List[Tuple[str, str, str]]) -> str:
        """
        Format relationships for inclusion in a prompt.

        Args:
            relationships: List of (source, relation, target) tuples

        Returns:
            str: Formatted relationship strings
        """
        if not relationships:
            return "None"

        formatted = []
        for source, relation, target in relationships:
            formatted.append(f"- {source} {relation} {target}")

        return "\n".join(formatted)

    def _parse_llm_response(self, response: str) -> float:
        """
        Parse the LLM response to get a similarity score.

        Args:
            response: The response from the LLM

        Returns:
            float: Similarity score (0-1)
        """
        try:
            # Extract a numeric value from the response
            import re

            numbers = re.findall(r"0\.\d+|\d+\.\d+|\d+", response)
            if numbers:
                score = float(numbers[0])
                return min(max(score, 0.0), 1.0)  # Ensure it's in range [0,1]
        except:  # noqa: E722
            pass

        return 0.0  # Default to no similarity if parsing fails

    def _convert_pairs_to_groups(
        self, pairs: List[Tuple[str, str]], nodes: List[str]
    ) -> List[Set[str]]:
        """
        Convert pairs of similar nodes to groups, handling transitive relationships.

        Args:
            pairs: List of (node1, node2) tuples indicating similar nodes
            nodes: List of all node names

        Returns:
            List[Set[str]]: List of sets where each set contains node names to be merged
        """
        # Use a Union-Find data structure to group nodes
        parent = {node: node for node in nodes}

        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x, y):
            parent[find(x)] = find(y)

        # Group nodes based on pairs
        for node1, node2 in pairs:
            union(node1, node2)

        # Collect groups
        groups = {}
        for node in nodes:
            root = find(node)
            if root not in groups:
                groups[root] = set()
            groups[root].add(node)

        # Filter to include only groups with multiple nodes
        return [group for group in groups.values() if len(group) > 1]
