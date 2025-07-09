import random
from typing import Callable, Dict, List, Optional, Set, Tuple

from langchain_core.prompts import PromptTemplate

from ardcore.knowledge_graph.knowledge_graph import KnowledgeGraph
from ardcore.subgraph.subgraph_generator.base import SingleNodeSubgraphGenerator


class LLMWalkGenerator(SingleNodeSubgraphGenerator):
    """
    Generates a subgraph by using an LLM to guide a walk from a start node.

    The LLM is provided context about the current path and all neighboring nodes
    at each step, and makes decisions about which node to visit next.
    """

    def __init__(
        self,
        llm: Callable[[str], str],
        max_steps: int = 10,
        prompt: Optional[PromptTemplate] = None,
        seed: Optional[int] = None,
    ):
        """
        Initialize the LLMWalkGenerator.

        Args:
            max_steps: Maximum number of steps for the walk
            llm: The language model to use for decision making
            prompt: Custom prompt template to use. If None, a default prompt will be used.
            seed: Random seed for reproducibility when falling back to random selection
        """
        self.max_steps = max_steps
        self.llm = llm
        self.prompt = prompt or DEFAULT_LLM_WALK_PROMPT
        self.seed = seed

        if seed is not None:
            random.seed(seed)

    def _format_path_for_llm(
        self, path: List[str], path_edges: List[Tuple[str, str, str]]
    ) -> str:
        """
        Format the current path for the LLM.

        Args:
            path: List of node names in the current path
            path_edges: List of (source, relation, target) tuples representing edges in the path

        Returns:
            str: Formatted path string for the LLM
        """
        if not path_edges:
            return f"Current path: {path[0]}"

        path_str = f"Current path: {path[0]}"
        for i, (source, relation, target) in enumerate(path_edges):
            # Check if this is a reverse relationship
            if relation.startswith("REVERSE_"):
                # Format as an incoming edge
                actual_relation = relation[8:]  # Remove the "REVERSE_" prefix
                path_str += f" <-[{actual_relation}]- {target}"
            elif relation == "unknown_relation":
                # Format as an unspecified connection
                path_str += f" -- {target}"
            else:
                # Format as a normal outgoing edge
                path_str += f" -[{relation}]-> {target}"

        return path_str

    def _format_neighbors_for_llm(
        self, knowledge_graph: KnowledgeGraph, current_node: str, visited: Set[str]
    ) -> Dict[str, List[Tuple[str, str, str]]]:
        """
        Format the current node's neighbors for the LLM.

        Args:
            knowledge_graph: The knowledge graph
            current_node: The current node
            visited: Set of already visited nodes

        Returns:
            Dict with 'unvisited' and 'visited' neighbors and their relations
        """
        # Separate visited and unvisited neighbors
        unvisited_neighbors = []
        visited_neighbors = []

        # Process outgoing edges
        for neighbor in knowledge_graph.get_successors(current_node, n_of_nodes=100):
            # Get edge data for this connection
            edges_data = knowledge_graph.get_edge_attrs(current_node, neighbor)

            for edge_data in edges_data:
                # Get the relation - handle different edge data formats
                if isinstance(edge_data, dict):
                    # Direct edge data
                    relation = edge_data.get("relation", "")
                else:
                    # Handle case where edge_data might be a string or other format
                    relation = str(edge_data) if edge_data else ""

                direction = "outgoing"

                if neighbor in visited:
                    visited_neighbors.append((neighbor, relation, direction))
                else:
                    unvisited_neighbors.append((neighbor, relation, direction))

        # Process incoming edges
        for neighbor in knowledge_graph.get_predecessors(current_node, n_of_nodes=100):
            # Skip if it's already a successor (we already processed it above)
            if knowledge_graph.has_edge(current_node, neighbor):
                continue

            # Get edge data for this incoming connection
            edges_data = knowledge_graph.get_edge_attrs(neighbor, current_node)

            for edge_data in edges_data:
                # Get the relation - handle different edge data formats
                if isinstance(edge_data, dict):
                    # Direct edge data
                    relation = edge_data.get("relation", "")
                else:
                    # Handle case where edge_data might be a string or other format
                    relation = str(edge_data) if edge_data else ""

                direction = "incoming"

                if neighbor in visited:
                    visited_neighbors.append((neighbor, relation, direction))
                else:
                    unvisited_neighbors.append((neighbor, relation, direction))

        return {"unvisited": unvisited_neighbors, "visited": visited_neighbors}

    def _format_neighbors_string(
        self, neighbors: Dict[str, List[Tuple[str, str, str]]]
    ) -> str:
        """
        Format the neighbors dictionary into a string for the LLM.

        Args:
            neighbors: Dictionary with 'unvisited' and 'visited' neighbors

        Returns:
            str: Formatted neighbors string
        """
        neighbor_str = "Available neighbors:\n"

        if neighbors["unvisited"]:
            neighbor_str += "Neighbors not yet visited:\n"
            for i, (node, relation, direction) in enumerate(neighbors["unvisited"]):
                if direction == "outgoing":
                    neighbor_str += f"{i + 1}. {node} (outgoing edge: -{relation}->)\n"
                else:
                    neighbor_str += f"{i + 1}. {node} (incoming edge: <-{relation}-)\n"
        else:
            neighbor_str += "No unvisited neighbors available.\n"

        if neighbors["visited"]:
            neighbor_str += "\nNeighbors already visited (can be revisited):\n"
            for i, (node, relation, direction) in enumerate(neighbors["visited"]):
                if direction == "outgoing":
                    neighbor_str += f"{i + 1}. {node} (outgoing edge: -{relation}->)\n"
                else:
                    neighbor_str += f"{i + 1}. {node} (incoming edge: <-{relation}-)\n"

        return neighbor_str

    def _get_next_node_from_llm(
        self,
        path_str: str,
        neighbors_str: str,
        neighbors: Dict[str, List[Tuple[str, str, str]]],
        current_node: str,
        goal: str = "Explore the graph in the most meaningful way",
    ) -> Optional[str]:
        """
        Ask the LLM to choose the next node to visit.

        Args:
            path_str: String representation of the current path
            neighbors_str: String representation of the available neighbors
            neighbors: Dictionary with 'unvisited' and 'visited' neighbors
            current_node: The current node
            goal: Optional goal description for the walk

        Returns:
            Optional[str]: The next node to visit, or None if no valid choice
        """
        # If there are no neighbors at all, return None
        if not neighbors["unvisited"] and not neighbors["visited"]:
            return None

        # Handle special case for isolated nodes with self-loops
        # If the only neighbor is the current node itself (self-loop), return None to avoid infinite loops
        if (len(neighbors["unvisited"]) + len(neighbors["visited"])) == 1:
            if (
                neighbors["unvisited"] and neighbors["unvisited"][0][0] == current_node
            ) or (neighbors["visited"] and neighbors["visited"][0][0] == current_node):
                return None

        # Prepare the prompt
        prompt_text = self.prompt.format(
            current_node=current_node, path=path_str, neighbors=neighbors_str, goal=goal
        )

        # Call the LLM
        response = self.llm(prompt_text)

        # Extract the content
        if hasattr(response, "content"):
            content = response.content
        else:
            content = str(response)

        # Parse the response to extract the node choice
        # Look for a line with "NEXT_NODE:" or "Next node:"
        import re

        node_match = re.search(r"(?:NEXT_NODE:|Next node:)\s*(\w+)", content)

        if node_match:
            next_node = node_match.group(1).strip()

            # Get all valid neighbor nodes (both unvisited and visited)
            all_neighbors = [node for node, _, _ in neighbors["unvisited"]] + [
                node for node, _, _ in neighbors["visited"]
            ]

            # Check if the chosen node is valid
            if next_node in all_neighbors:
                return next_node

        # If no valid choice was found, fall back to random selection
        # Prioritize unvisited neighbors for random selection, but allow visited if no unvisited are available
        if neighbors["unvisited"]:
            return random.choice([node for node, _, _ in neighbors["unvisited"]])
        elif neighbors["visited"]:
            return random.choice([node for node, _, _ in neighbors["visited"]])
        else:
            return None

    def _build_path_edges(
        self, path: List[str], knowledge_graph: KnowledgeGraph
    ) -> List[Tuple[str, str, str]]:
        """
        Build a list of path edges from a list of nodes.

        Args:
            path: List of node names
            knowledge_graph: The knowledge graph

        Returns:
            List of (source, relation, target) tuples
        """
        path_edges = []

        for i in range(len(path) - 1):
            source = path[i]
            target = path[i + 1]

            # Find the most appropriate edge between consecutive nodes
            # First, check direct outgoing edges
            if knowledge_graph.has_edge(source, target):
                # Get edge data
                edges_data = knowledge_graph.get_edge_attrs(source, target)

                for edge_data in edges_data:
                    # Handle different edge data formats
                    if isinstance(edge_data, dict):
                        # Get relation from the edge data dictionary
                        relation = edge_data.get("relation", "")
                    else:
                        # Handle case where edge_data might be a string or other format
                        relation = str(edge_data) if edge_data else ""

                    path_edges.append((source, relation, target))

            # If no direct edge, check for incoming edges
            elif knowledge_graph.has_edge(target, source):
                # Get edge data
                edges_data = knowledge_graph.get_edge_attrs(target, source)

                for edge_data in edges_data:
                    # Handle different edge data formats
                    if isinstance(edge_data, dict):
                        # Get relation from the edge data dictionary
                        relation = edge_data.get("relation", "")
                    else:
                        # Handle case where edge_data might be a string or other format
                        relation = str(edge_data) if edge_data else ""

                    # Mark that this is a reverse relationship
                    path_edges.append((source, f"REVERSE_{relation}", target))

            # If no direct edge in either direction, note that as "unknown_relation"
            else:
                path_edges.append((source, "unknown_relation", target))

        return path_edges

    def generate_path_nodes(
        self, knowledge_graph: KnowledgeGraph, start_node: str
    ) -> List[str]:
        """
        Generate a list of nodes from a walk starting at start_node,
        using an LLM to make decisions about which node to visit next.
        Nodes can be revisited, allowing for cycles in the path.

        Args:
            knowledge_graph: The KnowledgeGraph instance
            start_node: The starting node for the path

        Returns:
            List[str]: List of node names that form the path
        """
        # Validate the start node
        self.validate_node(knowledge_graph, start_node)

        # Start the walk
        path = [start_node]
        visited = {
            start_node
        }  # We still track visited nodes to display them differently
        current_node = start_node
        next_node = None

        for _ in range(self.max_steps):
            # Get neighbors
            neighbors = self._format_neighbors_for_llm(
                knowledge_graph, current_node, visited
            )

            # neighbors["unvisited"] = [
            #    n for n in neighbors["unvisited"] if n[0] != "Psilocybin"
            # ]

            # If there are no neighbors at all, we're done
            if not neighbors["unvisited"] and not neighbors["visited"]:
                break

            # Format the path and neighbors for the LLM
            path_edges = self._build_path_edges(path, knowledge_graph)
            path_str = self._format_path_for_llm(path, path_edges)
            neighbors_str = self._format_neighbors_string(neighbors)

            # Get the next node from the LLM
            next_node = self._get_next_node_from_llm(
                path_str, neighbors_str, neighbors, current_node
            )

            # If no next node was selected, we're done
            if next_node is None:
                break

            # Add the next node to the path (even if already visited)
            path.append(next_node)
            # Mark this node as visited (if not already)
            visited.add(next_node)
            current_node = next_node

        return path


# Default prompt template for the LLM
DEFAULT_LLM_WALK_PROMPT = PromptTemplate.from_template(
    """You are navigating through a knowledge graph. Your task is to choose the next node to visit.

Current node: {current_node}

{path}

{neighbors}

Goal: {goal}

Based on the available neighbors, which node should we visit next?
You can choose from any neighbor, including ones that have been visited before.
Choose the node that would provide the most meaningful path through the graph for hypothesis generation and knowledge discovery.

IMPORTANT: Prioritize nodes that represent substantive concepts, findings, or phenomena that could lead to new insights or hypotheses.
Avoid nodes that represent:
- Methodologies or research methods
- Measurement tools or instruments
- Technical procedures
- Administrative or generic concepts
- Nodes that don't contribute to novel scientific understanding

You can revisit nodes if that would create a more insightful exploration pattern.

NEXT_NODE: """
)
