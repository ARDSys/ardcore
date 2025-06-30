import hashlib
import json
import os
import random
import re
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import matplotlib.pyplot as plt
import networkx as nx
from langchain_core.prompts import PromptTemplate
from loguru import logger

from ard.knowledge_graph.knowledge_graph import KnowledgeGraph
from ard.storage.file import StorageBackend
from ard.subgraph.subgraph_generator import (
    SingleNodeSubgraphGenerator,
    SubgraphGenerator,
)


def select_additional_nodes(
    original_graph: KnowledgeGraph,
    path_nodes: List[str],
    neighbor_probability: float = 1.0,
    max_nodes: Optional[int] = None,
) -> Set[str]:
    """
    Select additional nodes to include in a subgraph based on probability and max nodes.

    Args:
        original_graph (KnowledgeGraph): The original knowledge graph
        path_nodes (List[str]): The nodes on the path from start to end
        neighbor_probability (float): Probability (0.0 to 1.0) of including each neighbor outside the path
        max_nodes (Optional[int]): Maximum number of nodes to include in the subgraph (None for no limit)

    Returns:
        Set[str]: Set of additional nodes to include in the subgraph
    """
    # Validate neighbor_probability
    if not 0.0 <= neighbor_probability <= 1.0:
        raise ValueError("neighbor_probability must be between 0.0 and 1.0")

    if max_nodes is not None and len(path_nodes) > max_nodes:
        # If path nodes already exceed max_nodes, don't add any additional nodes
        return set()

    # Collect all potential neighbors
    potential_neighbors = set()
    # Add neighbors based on probability
    if neighbor_probability > 0:
        for node in path_nodes:
            # Add all neighbors (both incoming and outgoing)
            neighbors = set(original_graph.get_node_neighbors(node))

            # Exclude nodes that are already in the path
            neighbors -= set(path_nodes)
            potential_neighbors.update(neighbors)

    # Start with an empty set of additional nodes
    additional_nodes = set()

    # Randomly select neighbors based on probability
    for neighbor in potential_neighbors:
        if random.random() < neighbor_probability:
            additional_nodes.add(neighbor)

    if max_nodes:
        remaining_slots = max_nodes - len(path_nodes)
    else:
        remaining_slots = len(additional_nodes)

    # Randomly sample the remaining nodes
    remaining_nodes = list(additional_nodes)
    random.shuffle(remaining_nodes)

    return set(remaining_nodes[:remaining_slots])


class Subgraph(KnowledgeGraph):
    """
    A subgraph extracted from a KnowledgeGraph.

    This class represents a subgraph containing a path between two nodes
    and their nearest neighbors.

    Attributes:
        _original_graph (KnowledgeGraph): The original knowledge graph
        _start_node (str): The starting node for the path
        _end_node (str): The ending node for the path
        _path_nodes (List[str]): The nodes on the path from start to end
        _subgraph_id (str): The unique ID for the subgraph
        _additional_nodes (Optional[Set[str]]): Additional nodes to include in the subgraph
    """

    def __init__(
        self,
        original_graph: KnowledgeGraph,
        start_node: str,
        end_node: str,
        path_nodes: List[str],
        additional_nodes: Optional[Set[str]] = None,
    ) -> None:
        """
        Initialize a Subgraph from a KnowledgeGraph and two nodes.

        Args:
            original_graph (KnowledgeGraph): The original knowledge graph
            start_node (str): The starting node for the path
            end_node (str): The ending node for the path
            path_nodes (List[str]): Nodes on the path from start_node to end_node
            additional_nodes (Optional[Set[str]]): Additional nodes to include in the subgraph

        Raises:
            ValueError: If start_node or end_node are not in the graph
            ValueError: If any path node is not in the graph
        """
        super().__init__(
            config=original_graph.config,
            scientific_domain=original_graph.scientific_domain,
        )

        self._original_graph = original_graph
        self._start_node = start_node
        self._end_node = end_node
        self._path_nodes = path_nodes
        self._context = None
        self._path_score = None
        self._path_score_justification = None

        # Generate a deterministic ID based on subgraph content
        self._additional_nodes = additional_nodes or set()
        self._subgraph_id = self._generate_subgraph_id()

        # Validate nodes exist in the graph
        if not original_graph.has_node(start_node):
            raise ValueError(f"Start node '{start_node}' not found in the graph")
        if not original_graph.has_node(end_node):
            raise ValueError(f"End node '{end_node}' not found in the graph")

        if any([not original_graph.has_node(n) for n in path_nodes]):
            raise ValueError(f"Path nodes {path_nodes} not found in the graph")

        # Extract the subgraph
        self._extract_subgraph(self._additional_nodes)

    def _generate_subgraph_id(self) -> str:
        """
        Generate a deterministic ID based on the content of the subgraph.

        The ID is a SHA-256 hash of a deterministic representation of:
        - Start node
        - End node
        - Path nodes
        - Additional nodes

        Returns:
            str: A hexadecimal digest of the hash
        """
        # Create a dictionary with all the elements that define the subgraph
        subgraph_elements = {
            "start_node": self._start_node,
            "end_node": self._end_node,
            "path_nodes": sorted(self._path_nodes),
            "additional_nodes": sorted(list(self._additional_nodes)),
        }

        # Convert to a deterministic JSON string
        json_str = json.dumps(subgraph_elements, sort_keys=True)

        # Hash the string and return the hexadecimal digest
        return hashlib.sha256(json_str.encode()).hexdigest()

    def _extract_subgraph(self, additional_nodes: Set[str]) -> None:
        """
        Extract the subgraph from the original graph.

        Args:
            additional_nodes (Set[str]): Additional nodes to include in the subgraph
        """
        # Nodes to include in the subgraph - always include path nodes
        nodes_to_include = set(self._path_nodes) | additional_nodes

        # Create a new knowledge graph for the subgraph by adding nodes and edges
        self._initialize_from_nodes_and_edges(
            original_graph=self._original_graph, nodes_to_include=nodes_to_include
        )

    def _initialize_from_nodes_and_edges(
        self, original_graph: KnowledgeGraph, nodes_to_include: set
    ) -> None:
        """
        Initialize this knowledge graph from a set of nodes and their edges.

        Args:
            original_graph: The original knowledge graph
            nodes_to_include: Set of node names to include
        """
        # Add all nodes from the original graph
        for node in nodes_to_include:
            # Add the node to our graph with its attributes
            attrs = original_graph.get_node_attrs(node)
            self.add_node(node, **attrs)

        # Add edges between nodes that are in our set
        for source in nodes_to_include:
            # Get all edges from the original graph
            for target in original_graph.get_successors(source):
                if target in nodes_to_include:
                    # Get edge attributes
                    edges_attrs = original_graph.get_edge_attrs(source, target)
                    for edge_attrs in edges_attrs:
                        # Add the edge to our graph
                        self.add_edge(source, target, **edge_attrs)

    @classmethod
    def from_two_nodes(
        cls,
        original_graph: KnowledgeGraph,
        start_node: str,
        end_node: str,
        method: SubgraphGenerator,
        neighbor_probability: float = 1.0,
        max_nodes: Optional[int] = None,
    ) -> "Subgraph":
        """
        Create a Subgraph between two nodes using the specified subgraph generation method.

        Args:
            original_graph (KnowledgeGraph): The original knowledge graph
            start_node (str): The starting node for the path
            end_node (str): The ending node for the path
            method (SubgraphGenerator): The subgraph generation method to use
            neighbor_probability (float): Probability of including each neighbor outside the path
            max_nodes (Optional[int]): Maximum number of nodes to include in the subgraph

        Returns:
            Subgraph: The subgraph
        """
        # Generate the path nodes using the provided method
        path_nodes = method.generate_path_nodes(original_graph, start_node, end_node)

        # Select additional nodes using the helper function
        additional_nodes = select_additional_nodes(
            original_graph, path_nodes, neighbor_probability, max_nodes
        )

        # Create and return the subgraph
        return cls(
            original_graph,
            start_node,
            end_node,
            path_nodes,
            additional_nodes,
        )

    @classmethod
    def from_one_node(
        cls,
        original_graph: KnowledgeGraph,
        start_node: str,
        method: SingleNodeSubgraphGenerator,
        neighbor_probability: float = 1.0,
        max_nodes: Optional[int] = None,
    ) -> "Subgraph":
        """
        Create a Subgraph starting from a single node using the specified subgraph generation method.

        Args:
            original_graph (KnowledgeGraph): The original knowledge graph
            start_node (str): The starting node for the path
            method (SingleNodeSubgraphGenerator): The single-node subgraph generation method to use
            neighbor_probability (float): Probability of including each neighbor outside the path
            max_nodes (Optional[int]): Maximum number of nodes to include in the subgraph

        Returns:
            Subgraph: The subgraph
        """
        # Generate the path nodes using the provided method
        path_nodes = method.generate_path_nodes(original_graph, start_node)

        # Ensure there's at least one node in the path
        if not path_nodes:
            raise ValueError(f"Empty path generated from node '{start_node}'")

        # Use the last node in the path as the end node
        end_node = path_nodes[-1]

        # Select additional nodes using the helper function
        additional_nodes = select_additional_nodes(
            original_graph, path_nodes, neighbor_probability, max_nodes
        )

        # Create and return the subgraph
        return cls(
            original_graph,
            start_node,
            end_node,
            path_nodes,
            additional_nodes,
        )

    @property
    def path_nodes(self) -> List[str]:
        """
        Get the nodes on the path from start to end.

        Returns:
            List[str]: The nodes on the path
        """
        return self._path_nodes

    @property
    def start_node(self) -> str:
        """
        Get the starting node.

        Returns:
            str: The starting node
        """
        return self._start_node

    @property
    def end_node(self) -> str:
        """
        Get the ending node.

        Returns:
            str: The ending node
        """
        return self._end_node

    @property
    def subgraph_id(self) -> str:
        """
        Get the subgraph ID.

        Returns:
            str: The subgraph ID
        """
        return self._subgraph_id

    @property
    def original_graph(self) -> KnowledgeGraph:
        """
        Get the original knowledge graph.

        Returns:
            KnowledgeGraph: The original knowledge graph
        """
        return self._original_graph

    @property
    def context(self) -> Optional[str]:
        """
        Get the analysis context of the subgraph.

        Returns:
            Optional[str]: The analysis context if it has been generated, None otherwise
        """
        return self._context

    def get_path_edges(self) -> List[Tuple[str, str, str]]:
        """
        Get the edges on the path from start to end.

        Returns:
            List[Tuple[str, str, str]]: List of (source, relation, target) tuples on the path
        """
        path_edges = []
        for i in range(len(self._path_nodes) - 1):
            source = self._path_nodes[i]
            target = self._path_nodes[i + 1]

            # Check if there's a direct edge in the graph
            if self.has_edge(source, target):
                edges_attrs = self.get_edge_attrs(source, target)
                for edge_attrs in edges_attrs:
                    relation = edge_attrs.get("relation", "")
                    path_edges.append((source, relation, target))
            elif self.has_edge(target, source):
                # If the edge is in the reverse direction
                edges_attrs = self.get_edge_attrs(target, source)
                for edge_attrs in edges_attrs:
                    relation = edge_attrs.get("relation", "")
                    path_edges.append((target, relation, source))

        return path_edges

    def __str__(self) -> str:
        """
        Get a string representation of the subgraph.

        Returns:
            str: A string representation
        """
        subgraph_str = f'Subgraph(start="{self._start_node}", end="{self._end_node}", path_length={len(self._path_nodes)})'
        base_str = super().__str__()
        path_str = " -> ".join(self._path_nodes)
        return f"{subgraph_str}\n{base_str}\nPath: {path_str}"

    def visualize(
        self,
        figsize: Tuple[int, int] = (12, 8),
        node_size: int = 1000,
        font_size: int = 10,
        edge_label_font_size: int = 8,
        save_path: Optional[str] = None,
        title: Optional[str] = None,
        layout: Optional[str] = None,
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Visualize the subgraph using NetworkX and matplotlib.

        Args:
            figsize: Tuple of (width, height) for the figure
            node_size: Size of the nodes
            font_size: Font size for node labels
            edge_label_font_size: Font size for edge labels
            save_path: Path to save the visualization (if None, display instead)
            title: Title for the visualization
            layout: Layout algorithm to use (if None, spring_layout is used)
                   Can be 'spring', 'circular', 'kamada_kawai', 'planar', 'random', 'shell', 'spectral'
                   or a dictionary mapping nodes to positions

        Returns:
            matplotlib figure and axes objects
        """
        # Create figure and axis
        fig, ax = plt.subplots(figsize=figsize)

        # For visualization, we need to access the NetworkX graph directly
        # This is a safe operation as we're just visualizing, not modifying
        backend_graph = self._backend.graph

        # Choose layout algorithm
        if layout is None or layout == "spring":
            pos = nx.spring_layout(backend_graph)
        elif layout == "circular":
            pos = nx.circular_layout(backend_graph)
        elif layout == "kamada_kawai":
            pos = nx.kamada_kawai_layout(backend_graph)
        elif layout == "planar":
            try:
                pos = nx.planar_layout(backend_graph)
            except nx.NetworkXException:
                pos = nx.spring_layout(backend_graph)
                print("Graph is not planar. Using spring layout instead.")
        elif layout == "random":
            pos = nx.random_layout(backend_graph)
        elif layout == "shell":
            pos = nx.shell_layout(backend_graph)
        elif layout == "spectral":
            pos = nx.spectral_layout(backend_graph)
        elif isinstance(layout, dict):
            pos = layout
        else:
            pos = nx.spring_layout(backend_graph)

        # Get all nodes except start and end
        regular_nodes = [
            node
            for node in self.get_nodes()
            if node != self._start_node and node != self._end_node
        ]

        # Draw regular nodes
        nx.draw_networkx_nodes(
            backend_graph,
            pos,
            nodelist=regular_nodes,
            node_color="skyblue",
            node_size=node_size,
            alpha=0.8,
            ax=ax,
        )

        # Draw start node in green
        if self.has_node(self._start_node):
            nx.draw_networkx_nodes(
                backend_graph,
                pos,
                nodelist=[self._start_node],
                node_color="green",
                node_size=node_size,
                alpha=0.8,
                ax=ax,
            )

        # Draw end node in red
        if self.has_node(self._end_node):
            nx.draw_networkx_nodes(
                backend_graph,
                pos,
                nodelist=[self._end_node],
                node_color="red",
                node_size=node_size,
                alpha=0.8,
                ax=ax,
            )

        # Draw edges
        nx.draw_networkx_edges(
            backend_graph,
            pos,
            width=1.5,
            alpha=0.7,
            arrowsize=20,
            ax=ax,
        )

        # Draw node labels
        nx.draw_networkx_labels(
            backend_graph,
            pos,
            font_size=font_size,
            font_weight="bold",
            ax=ax,
        )

        # Draw edge labels (relations)
        edge_labels = {}
        for u, v in backend_graph.edges():
            edges_attrs = self.get_edge_attrs(u, v)
            for i, edge_attrs in enumerate(edges_attrs):
                edge_labels[(u, v, i)] = edge_attrs.get("relation", "")

        nx.draw_networkx_edge_labels(
            backend_graph,
            pos,
            edge_labels=edge_labels,
            font_size=edge_label_font_size,
            ax=ax,
        )

        # Highlight the shortest path
        path_edges = list(zip(self._path_nodes[:-1], self._path_nodes[1:]))
        nx.draw_networkx_edges(
            backend_graph,
            pos,
            edgelist=path_edges,
            width=3.0,
            alpha=1.0,
            edge_color="purple",
            ax=ax,
        )

        # Set title
        if title:
            plt.title(title, fontsize=16)
        else:
            plt.title(
                f"Subgraph from {self._start_node} to {self._end_node}", fontsize=16
            )

        # Remove axis
        plt.axis("off")

        # Tight layout
        plt.tight_layout()

        # Save or show
        if save_path:
            plt.savefig(save_path, bbox_inches="tight", dpi=300)

        return fig, ax

    def _escape_pattern(self, pattern: str) -> str:
        """
        Escape any special characters in the pattern.
        """
        return f"`{pattern}`" if any(c in pattern for c in " -,()[]{}") else pattern

    def _edge_to_cypher(self, source: str, relation: str, target: str) -> str:
        """
        Convert an edge to Cypher notation.
        """
        source_escaped = self._escape_pattern(source)
        target_escaped = self._escape_pattern(target)
        relation_escaped = self._escape_pattern(relation)
        return f"({source_escaped})-[:{relation_escaped}]->({target_escaped})"

    def to_cypher_string(self, full_graph: bool = True) -> str:
        """
        Return a string representation of the subgraph using Cypher notation.

        The output format follows Neo4j's Cypher query language pattern:
        (node1)-[relation_type]->(node2)

        Each node and relationship is represented with its properties.
        The path from start to end node is listed first, followed by other edges.

        Args:
            full_graph (bool): Whether to include all edges in the graph. If False, only the path edges are included.

        Returns:
            str: A string containing Cypher representation of the subgraph with nodes and relationships
        """
        # Get the path edges first
        path_edges = self.get_path_edges()
        all_patterns = []

        # Process path edges
        for source, relation, target in path_edges:
            all_patterns.append(self._edge_to_cypher(source, relation, target))

        if full_graph:
            # Get all other edges that aren't in the path
            path_edge_pairs = {(s, t) for s, _, t in path_edges} | {
                (t, s) for s, _, t in path_edges
            }

            # Get all edges in the graph
            for source in self.get_nodes():
                for target in self.get_successors(source):
                    if (source, target) not in path_edge_pairs:
                        # Get edge attributes
                        edges_attrs = self.get_edge_attrs(source, target)
                        for edge_attrs in edges_attrs:
                            all_patterns.append(
                                self._edge_to_cypher(
                                    source, edge_attrs["relation"], target
                                )
                            )

        # Join patterns with commas and newlines for readability
        return ",\n".join(all_patterns)

    def score_path(
        self,
        llm: Callable[[str], str],
        prompt: Optional[PromptTemplate] = None,
    ) -> str:
        """
        Review the path of the subgraph and provide a score between 1 (very bad) and 5 (very good).

        Args:
            llm: The language model to use for scoring. If None, will attempt to use a default model.
            prompt: The prompt to use for scoring. If None, will use the default prompt.
                   Available keys: graph_str, start_node, end_node, scientific_domain.

        Returns:
            str: The score of the path.
        """
        # Import here to avoid circular imports
        from ard.subgraph.prompts import SUBGRAPH_SCORE_PROMPT

        # Use default prompt if none provided
        if prompt is None:
            prompt = SUBGRAPH_SCORE_PROMPT

        # Create a representation of the graph for the LLM
        graph_str = self.to_cypher_string(full_graph=False)

        # Generate the analysis
        final_prompt = prompt.format(
            graph_str=graph_str,
            start_node=self._start_node,
            end_node=self._end_node,
            scientific_domain=self.scientific_domain,
        )

        # Invoke the LLM
        analysis = llm(final_prompt)

        # Get the content (handle both string and message-like objects)
        if hasattr(analysis, "content"):
            content = analysis.content
        else:
            content = str(analysis)

        # Extract the score and justification
        score_match = re.search(r"rating=(\d+)", content)
        if score_match:
            self._path_score = int(score_match.group(1))
            self._path_score_justification = re.sub(r"rating=\d+", "", content).strip()

        return content

    def contextualize(
        self,
        llm: Callable[[str], str],
        prompt: Optional[PromptTemplate] = None,
    ) -> str:
        """
        Generate an analysis of concepts and relationships in the subgraph using an LLM.

        Args:
            llm: The language model to use for analysis. If None, will attempt to use a default model.
            prompt: The prompt to use for analysis. If None, will use the default prompt.
                   Available keys: graph_str, start_node, end_node, scientific_domain.

        Returns:
            str: The LLM's analysis of the subgraph's concepts and relationships.
        """
        # Import here to avoid circular imports
        from ard.subgraph.prompts import SUBGRAPH_ANALYSIS_PROMPT

        # Use default prompt if none provided
        if prompt is None:
            prompt = SUBGRAPH_ANALYSIS_PROMPT

        # Create a representation of the graph for the LLM
        graph_str = self.to_cypher_string()

        # Generate the analysis
        final_prompt = prompt.format(
            graph_str=graph_str,
            start_node=self._start_node,
            end_node=self._end_node,
            scientific_domain=self.scientific_domain,
        )

        # Invoke the LLM
        analysis = llm(final_prompt)

        # Get the content (handle both string and message-like objects)
        if hasattr(analysis, "content"):
            self._context = analysis.content
            return analysis.content

        self._context = str(analysis)
        return str(analysis)

    def save_to_file(
        self,
        filename: str,
        storage: Optional[StorageBackend] = None,
        item_id: str = "subgraphs",
    ) -> None:
        """
        Save the subgraph to a file.

        This method saves the graph structure and the subgraph-specific attributes
        like start node, end node, path nodes, and context. For the original graph,
        only essential metadata is stored rather than the entire graph.

        Args:
            filename: Path to the file where the subgraph will be saved
            storage: Optional storage backend. If None, saves to local filesystem

        Raises:
            IOError: If the file cannot be written
        """
        # Convert to JSON format
        data = self.to_json()

        if "timestamp" not in data:
            data["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())

        if storage:
            # Serialize to UTF-8 encoded bytes and save via storage backend
            json_data = json.dumps(data, indent=2, ensure_ascii=False).encode("utf-8")
            logger.info("Saving JSON file via storage backend")
            storage.save_file(item_id, str(filename), json_data)
        else:
            # Handle the path relative to current directory
            path = Path(filename).resolve()  # Get absolute path
            logger.info("Saving JSON file to local filesystem")
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

    def to_json(self) -> Dict[str, Any]:
        """
        Convert the subgraph metadata and scores to a JSON-serializable dictionary.

        This method exports the human-readable metadata about the subgraph
        without including the actual graph structure itself.

        Returns:
            Dict[str, Any]: Dictionary containing metadata and scores
        """
        # Get path edges for the JSON representation
        path_edges = []
        for i in range(len(self._path_nodes) - 1):
            source = self._path_nodes[i]
            target = self._path_nodes[i + 1]
            relation = ""

            # Check if there's a direct edge in the graph
            if self.has_edge(source, target):
                edges_attrs = self.get_edge_attrs(source, target)
                for edge_attrs in edges_attrs:
                    relation = edge_attrs.get("relation", "")
            elif self.has_edge(target, source):
                edges_attrs = self.get_edge_attrs(target, source)
                for edge_attrs in edges_attrs:
                    relation = edge_attrs.get("relation", "")

            path_edges.append(
                {"source": source, "target": target, "relation": relation}
            )

        # Add the graph structure data
        graph_data = {
            "nodes": {node: self.get_node_attrs(node) for node in self.get_nodes()},
            "edges": [
                {"source": source, "target": target, **edge_attr}
                for source in self.get_nodes()
                for target in self.get_successors(source)
                for edge_attr in (
                    self.get_edge_attrs(source, target)
                    if isinstance(self.get_edge_attrs(source, target), list)
                    else [self.get_edge_attrs(source, target)]
                )
            ],
        }

        # Extract graph statistics
        graph_stats = {
            "node_count": self.number_of_nodes(),
            "edge_count": self.number_of_edges(),
            "path_length": len(self._path_nodes),
        }

        # Extract original graph metadata
        original_graph_metadata = {
            "node_count": self._original_graph.number_of_nodes(),
            "edge_count": self._original_graph.number_of_edges(),
        }

        # Build the complete metadata dictionary
        metadata = {
            "subgraph_id": self._subgraph_id,
            "graph_data": graph_data,
            "graph_stats": graph_stats,
            "start_node": self._start_node,
            "end_node": self._end_node,
            "path_nodes": self._path_nodes,
            "path_edges": path_edges,
            "additional_nodes": list(self._additional_nodes),
            "context": self._context,
            "path_score": self._path_score,
            "path_score_justification": self._path_score_justification,
            "original_graph_metadata": original_graph_metadata,
            "config": self.config,
        }

        return metadata

    @classmethod
    def load_from_file(
        cls,
        filename: str,
        scientific_domain: str,
        storage: Optional[StorageBackend] = None,
        item_id: Optional[str] = "subgraphs",
    ) -> "Subgraph":
        """
        Load a subgraph from a file.

        Args:
            filename: Path to the file containing the saved subgraph

        Returns:
            Subgraph: The loaded subgraph

        Raises:
            FileNotFoundError: If the file does not exist
            IOError: If the file cannot be read
            ValueError: If the file does not contain a valid subgraph
        """
        if storage:
            # Load from storage backend
            json_data = storage.get_file(item_id, str(filename))
            data = json.loads(json_data)
        else:
            # Load from local filesystem
            if not os.path.exists(filename):
                raise FileNotFoundError(f"File not found: {filename}")

            # Load data from file
            with open(filename, "r", encoding="utf-8") as f:
                data = json.load(f)

        # Create a new instance without initialization
        subgraph = cls.__new__(cls)

        # Set basic attributes from loaded data
        subgraph.config = data.get("config")
        subgraph._start_node = data["start_node"]
        subgraph._end_node = data["end_node"]
        subgraph._path_nodes = data["path_nodes"]
        subgraph._context = data["context"]
        subgraph._path_score = data.get("path_score")
        subgraph._path_score_justification = data.get("path_score_justification")
        subgraph.scientific_domain = scientific_domain

        # Handle additional nodes (might not be present in older saved files)
        additional_nodes_data = data.get("additional_nodes", [])
        subgraph._additional_nodes = (
            set(additional_nodes_data) if additional_nodes_data else set()
        )

        # Generate the subgraph ID or use the stored one
        stored_id = data.get("subgraph_id")
        subgraph._subgraph_id = stored_id

        # Create a minimal KnowledgeGraph to represent the original graph
        original_graph = KnowledgeGraph(
            config=data.get("original_graph_metadata", {}).get("config"),
            scientific_domain=scientific_domain,
        )

        # Create a new backend for the subgraph
        from ard.storage.graph import NetworkXBackend

        subgraph._backend = NetworkXBackend()

        # Reconstruct the graph structure
        graph_data = data["graph_data"]

        # Add nodes
        for node, attrs in graph_data["nodes"].items():
            subgraph.add_node(node, **attrs)
            original_graph.add_node(node, **attrs)

        # Add edges
        for edge in graph_data["edges"]:
            source = edge.pop("source")
            target = edge.pop("target")
            subgraph.add_edge(source, target, **edge)
            original_graph.add_edge(source, target, **edge)

        subgraph._original_graph = original_graph

        # If loading an older file without the additional_nodes attribute,
        # regenerate the ID to ensure it's consistent
        if stored_id is None or not hasattr(subgraph, "_additional_nodes"):
            subgraph._additional_nodes = set()
            subgraph._subgraph_id = subgraph._generate_subgraph_id()

        return subgraph
