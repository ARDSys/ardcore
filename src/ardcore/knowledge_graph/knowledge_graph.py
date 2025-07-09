import json
import os
import random
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from loguru import logger

from ardcore.data.dataset import Dataset
from ardcore.data.triplets import Triplet, Triplets
from ardcore.knowledge_graph.node_merger import NodeMerger
from ardcore.storage.graph import GraphBackend, Neo4jBackend, NetworkXBackend


class KnowledgeGraph:
    """
    A knowledge graph representation.

    This class provides a graph-focused interface for working with knowledge
    represented as a directed graph, where edges can have relation types and metadata.

    Attributes:
        _backend (GraphBackend): The graph backend implementation
        config (Dict): Configuration parameters for the knowledge graph
    """

    def __init__(
        self,
        scientific_domain: str,
        config: Dict = None,
        backend: str = "networkx",
        **backend_config,
    ) -> None:
        """
        Initialize a KnowledgeGraph instance.

        Args:
            config (Dict, optional): Configuration parameters for the knowledge graph
            backend (str): The backend to use ("networkx" or "neo4j")
            **backend_config: Additional configuration for the backend
        """
        self.config = config or {}
        self.scientific_domain = scientific_domain

        # Initialize the appropriate backend
        if backend == "networkx":
            self._backend = NetworkXBackend()
        elif backend == "neo4j":
            self._backend = Neo4jBackend(
                uri=backend_config["uri"],
                user=backend_config["user"],
                password=backend_config["password"],
            )
        else:
            raise ValueError(f"Unsupported backend: {backend}")

    @classmethod
    def from_dataset(
        cls,
        dataset: Dataset,
        skip_errors: bool = True,
        max_items: Optional[int] = None,
        kg_version: Optional[str] = None,
        **kwargs,
    ) -> "KnowledgeGraph":
        """
        Create a KnowledgeGraph from a Dataset instance.

        Args:
            dataset (Dataset): The dataset to build the graph from
            max_items (Optional[int]): The maximum number of items to process
            **kwargs: Additional arguments to pass to the constructor
        """
        kg = cls(**kwargs)
        for item in dataset.items[:max_items]:
            try:
                triplets = item.get_triplets(kg_version=kg_version)
                kg.add_triplets(triplets)
            except Exception as e:
                if skip_errors:
                    logger.warning(f"Error processing item {item.id}: {e}")
                else:
                    raise e
        return kg

    @classmethod
    def from_triplets(
        cls,
        triplets: Union[List[Triplet], Triplets, List[Triplets]],
        **kwargs,
    ) -> "KnowledgeGraph":
        """
        Create a KnowledgeGraph from a list of Triplet objects or a Triplets instance.

        Args:
            triplets (Union[List[Triplet], Triplets, List[Triplets]]): The triplets to build the graph from
            **kwargs: Additional arguments to pass to the constructor

        Returns:
            KnowledgeGraph: A new KnowledgeGraph instance
        """
        kg = cls(**kwargs)

        # Extract triplet list from different input types
        if isinstance(triplets, Triplets):
            # Use the triplets from the Triplets instance
            triplet_list = triplets.triplets
        else:
            # If triplets is a list, flatten it if needed
            triplet_list = []
            for t in triplets:
                if isinstance(t, Triplet):
                    triplet_list.append(t)
                elif isinstance(t, Triplets):
                    triplet_list.extend(t.triplets)

        kg.add_triplets(triplet_list)

        return kg

    def add_triplets(self, triplets: Union[List[Triplet], Triplets]) -> None:
        """
        Add a list of triplets to the graph.

        Args:
            triplets (List[Triplet]): The triplets to add to the graph
        """
        if isinstance(triplets, Triplets):
            triplet_list = triplets.triplets
        else:
            triplet_list = triplets

        # Add all triplets to the graph
        for triplet in triplet_list:
            # Add source node if it doesn't exist
            if not self.has_node(triplet.node_1):
                self.add_node(triplet.node_1, sources=[])

            # Add target node if it doesn't exist
            if not self.has_node(triplet.node_2):
                self.add_node(triplet.node_2, sources=[])

            # Create source metadata from triplet metadata
            source_metadata = {
                "relation": triplet.edge,
                "triplet_id": id(triplet),  # Use object id as a unique identifier
            }
            if triplet.metadata:
                source_metadata.update(triplet.metadata)

            # Add metadata to source node's sources
            node1_attrs = self.get_node_attrs(triplet.node_1)
            node1_attrs["sources"].append(source_metadata.copy())
            self.add_node(triplet.node_1, **node1_attrs)

            # Add metadata to target node's sources
            node2_attrs = self.get_node_attrs(triplet.node_2)
            node2_attrs["sources"].append(source_metadata.copy())
            self.add_node(triplet.node_2, **node2_attrs)

            # Check if edge already exists
            if self.has_edge(triplet.node_1, triplet.node_2):
                # If edge exists, get its current attributes
                edges_data = self.get_edge_attrs(triplet.node_1, triplet.node_2)
                edge_data = {}

                for edge in edges_data:
                    if edge["relation"] == triplet.edge:
                        edge_data = edge
                        break

                # there are edges with other relations, so we need to add the new relation to the edge
                if edge_data == {}:
                    edge_data["relation"] = triplet.edge

                # Make sure sources list exists
                if "sources" not in edge_data:
                    edge_data["sources"] = []

                # Add this triplet's metadata to sources
                edge_data["sources"].append(source_metadata)

                self.add_edge(triplet.node_1, triplet.node_2, **edge_data)
            else:
                # Create a new edge with the relation and a sources list
                edge_data = {"relation": triplet.edge, "sources": [source_metadata]}
                self.add_edge(triplet.node_1, triplet.node_2, **edge_data)

    def add_node(self, node: str, **attrs) -> None:
        """
        Add a node with optional attributes.

        Args:
            node (str): The node identifier
            **attrs: Optional node attributes
        """
        # Remove scientific_domain from attrs to avoid passing scientific_domain two times
        attrs.pop("scientific_domain", None)
        self._backend.add_node(node, self.scientific_domain, **attrs)

    def has_node(self, node: str) -> bool:
        """
        Check if a node exists.

        Args:
            node (str): The node identifier

        Returns:
            bool: True if the node exists, False otherwise
        """
        return self._backend.has_node(node, self.scientific_domain)

    def add_edge(self, source: str, target: str, **attrs) -> None:
        """
        Add an edge with optional attributes.

        Args:
            source (str): The source node
            target (str): The target node
            **attrs: Optional edge attributes
        """
        # Remove scientific_domain from attrs to avoid passing scientific_domain two times
        attrs.pop("scientific_domain", None)
        self._backend.add_edge(
            source=source,
            target=target,
            scientific_domain=self.scientific_domain,
            **attrs,
        )

    def has_edge(self, source: str, target: str) -> bool:
        """
        Check if an edge exists.

        Args:
            source (str): The source node
            target (str): The target node

        Returns:
            bool: True if the edge exists, False otherwise
        """
        return self._backend.has_edge(
            source=source, target=target, scientific_domain=self.scientific_domain
        )

    def get_node_attrs(self, node: str) -> Dict[str, Any]:
        """
        Get all attributes of a node.

        Args:
            node (str): The node identifier

        Returns:
            Dict[str, Any]: Dictionary of node attributes
        """
        return self._backend.get_node_attrs(
            node, scientific_domain=self.scientific_domain
        )

    def get_edge_attrs(self, source: str, target: str) -> List[Dict[str, Any]]:
        """
        Get all attributes of an edge.

        Args:
            source (str): The source node
            target (str): The target node

        Returns:
            Dict[str, Any]: Dictionary of edge attributes
        """
        return self._backend.get_edge_attrs(
            source=source, target=target, scientific_domain=self.scientific_domain
        )

    def get_edges_data(self) -> List[Tuple[str, str, Dict[str, Any]]]:
        """
        Get all edges in the graph with their attributes.

        Returns:
            List[Tuple[str, str, Dict[str, Any]]]: List of (source, target, attributes) tuples
        """

        """
        edges = self._backend.get_edges()
    
        # Normalize key naming from 'edge' to 'relation'
        normalized_edges = []
        for source, target, attrs in edges:
            normalized_attrs = attrs.copy()
            if 'edge' in normalized_attrs and 'relation' not in normalized_attrs:
                normalized_attrs['relation'] = normalized_attrs.pop('edge')
            normalized_edges.append((source, target, normalized_attrs))
    
        return normalized_edges
        """

        return self._backend.get_edges(scientific_domain=self.scientific_domain)

    def get_successors(self, node: str, n_of_nodes: int = None) -> List[str]:
        """
        Get all successor nodes of a node.

        Args:
            node (str): The node identifier

        Returns:
            List[str]: List of successor node identifiers
        """
        return self._backend.get_successors(
            node, scientific_domain=self.scientific_domain, n_of_nodes=n_of_nodes
        )

    def get_predecessors(self, node: str, n_of_nodes: int = None) -> List[str]:
        """
        Get all predecessor nodes of a node.

        Args:
            node (str): The node identifier

        Returns:
            List[str]: List of predecessor node identifiers
        """
        return self._backend.get_predecessors(
            node, scientific_domain=self.scientific_domain, n_of_nodes=n_of_nodes
        )

    def get_out_edges(self, node: str) -> List[Tuple[str, str, Dict[str, Any]]]:
        """
        Get all outgoing edges of a node with their attributes.

        Args:
            node (str): The node identifier

        Returns:
            List[Tuple[str, str, Dict[str, Any]]]: List of (source, target, attributes) tuples
        """
        return self._backend.get_out_edges(
            node, scientific_domain=self.scientific_domain
        )

    def get_in_edges(self, node: str) -> List[Tuple[str, str, Dict[str, Any]]]:
        """
        Get all incoming edges of a node with their attributes.

        Args:
            node (str): The node identifier

        Returns:
            List[Tuple[str, str, Dict[str, Any]]]: List of (source, target, attributes) tuples
        """
        return self._backend.get_in_edges(
            node, scientific_domain=self.scientific_domain
        )

    def remove_node(self, node: str) -> None:
        """
        Remove a node and all its edges.

        Args:
            node (str): The node identifier
        """
        self._backend.remove_node(node, scientific_domain=self.scientific_domain)

    def number_of_edges(self) -> int:
        """
        Get the total number of edges in the graph.

        Returns:
            int: Number of edges
        """
        return self._backend.number_of_edges(scientific_domain=self.scientific_domain)

    def number_of_nodes(self) -> int:
        """
        Get the total number of nodes in the graph.

        Returns:
            int: Number of nodes
        """
        return len(self.get_nodes())

    @property
    def graph(self) -> GraphBackend:
        """
        Get the graph backend.

        Returns:
            GraphBackend: The graph backend implementation
        """
        return self._backend

    def has_graph(self) -> bool:
        """
        Check if the graph has been built.

        Returns:
            bool: True if the graph exists, False otherwise
        """
        return self._backend.__len__(self.scientific_domain) > 0

    def get_nodes(self) -> Set[str]:
        """
        Get all nodes in the graph.

        Returns:
            Set[str]: A set of all node names
        """
        return self._backend.get_nodes(scientific_domain=self.scientific_domain)

    def get_random_node(self) -> Optional[str]:
        """
        Get a random node from the graph.

        Returns:
            Optional[str]: A randomly selected node name, or None if the graph is empty
        """
        nodes = list(self.get_nodes())
        if not nodes:
            return None
        return random.choice(nodes)

    def get_edges(self) -> Set[str]:
        """
        Get all edge types in the graph.

        Returns:
            Set[str]: A set of all edge types
        """
        edge_types = set()

        for _, _, data in self.get_edges_data():
            edge_types.add(data["relation"])

        return edge_types

    def get_node_neighbors_relations(self, node: str) -> List[Tuple[str, str, str]]:
        """
        Get all neighbors of a node.

        Args:
            node (str): The node to get neighbors for

        Returns:
            List[Tuple[str, str, str]]: A list of (source, relation, target) tuples
        """
        if not self.has_node(node):
            return []

        neighbors = []

        # Outgoing edges
        for _, target, data in self.get_out_edges(node):
            neighbors.append((node, data.get("relation", ""), target))

        # Incoming edges
        for source, _, data in self.get_in_edges(node):
            neighbors.append((source, data.get("relation", ""), node))

        return neighbors

    def get_node_neighbors(self, node: str) -> List[str]:
        """
        Get all neighbors of a node, both incoming and outgoing.

        Args:
            node (str): The node to get neighbors for

        Returns:
            List[str]: A list of node names
        """
        if not self.has_node(node):
            return []

        neighbors = set(self.get_successors(node))
        neighbors.update(self.get_predecessors(node))
        return list(neighbors)

    def merge_nodes(self, node1: str, node2: str, merged_node: str) -> None:
        """
        Merge two nodes into a single node, combining their edges and metadata.

        Args:
            node1 (str): First node to merge
            node2 (str): Second node to merge
            merged_node (str): Name of the resulting merged node
        """
        if not self.has_node(node1) or not self.has_node(node2):
            return

        # Create the merged node if it doesn't exist yet
        if not self.has_node(merged_node):
            self.add_node(merged_node, sources=[])

        # Combine sources from both nodes
        node1_attrs = self.get_node_attrs(node1)
        node2_attrs = self.get_node_attrs(node2)
        merged_attrs = {"sources": []}

        if "sources" in node1_attrs:
            merged_attrs["sources"].extend(node1_attrs["sources"])
        if "sources" in node2_attrs:
            merged_attrs["sources"].extend(node2_attrs["sources"])

        self.add_node(merged_node, **merged_attrs)

        # Redirect all edges from node1 and node2 to merged_node
        # Outgoing edges from node1
        for _, target, data in self.get_out_edges(node1):
            if not self.has_edge(merged_node, target):
                # Create new edge with empty sources list
                self.add_edge(
                    merged_node, target, relation=data.get("relation", ""), sources=[]
                )

            # Add sources from this edge to the new edge
            if "sources" in data:
                edges_data = self.get_edge_attrs(merged_node, target)
                for edge_data in edges_data:
                    if isinstance(edge_data["sources"], str):
                        edge_data["sources"] = json.loads(edge_data["sources"])
                    if isinstance(data["sources"], str):
                        data["sources"] = json.loads(data["sources"])

                    edge_data["sources"].extend(data["sources"])
                    self.add_edge(merged_node, target, **edge_data)

        # Outgoing edges from node2
        for _, target, data in self.get_out_edges(node2):
            if not self.has_edge(merged_node, target):
                # Create new edge with empty sources list
                self.add_edge(
                    merged_node, target, relation=data.get("relation", ""), sources=[]
                )

            # Add sources from this edge to the new edge
            if "sources" in data:
                edges_data = self.get_edge_attrs(merged_node, target)
                for edge_data in edges_data:
                    if isinstance(edge_data["sources"], str):
                        edge_data["sources"] = json.loads(edge_data["sources"])
                    if isinstance(data["sources"], str):
                        data["sources"] = json.loads(data["sources"])

                    edge_data["sources"].extend(data["sources"])
                    self.add_edge(merged_node, target, **edge_data)

        # Incoming edges to node1
        for source, _, data in self.get_in_edges(node1):
            if not self.has_edge(source, merged_node):
                # Create new edge with empty sources list
                self.add_edge(
                    source, merged_node, relation=data.get("relation", ""), sources=[]
                )

            # Add sources from this edge to the new edge
            if "sources" in data:
                edges_data = self.get_edge_attrs(source, merged_node)
                for edge_data in edges_data:
                    if isinstance(edge_data["sources"], str):
                        edge_data["sources"] = json.loads(edge_data["sources"])
                    if isinstance(data["sources"], str):
                        data["sources"] = json.loads(data["sources"])

                    edge_data["sources"].extend(data["sources"])
                    self.add_edge(source, merged_node, **edge_data)

        # Incoming edges to node2
        for source, _, data in self.get_in_edges(node2):
            if not self.has_edge(source, merged_node):
                # Create new edge with empty sources list
                self.add_edge(
                    source, merged_node, relation=data.get("relation", ""), sources=[]
                )

            # Add sources from this edge to the new edge
            if "sources" in data:
                edge_data = self.get_edge_attrs(source, merged_node)
                if isinstance(edge_data["sources"], str):
                    edge_data["sources"] = json.loads(edge_data["sources"])
                if isinstance(data["sources"], str):
                    data["sources"] = json.loads(data["sources"])

                edge_data["sources"].extend(data["sources"])
                self.add_edge(source, merged_node, **edge_data)

        # Remove the original nodes
        self.remove_node(node1)
        self.remove_node(node2)

    def merge_similar_nodes(self, merger: NodeMerger) -> None:
        """
        Find and merge similar nodes using the provided merger strategy.

        Args:
            merger (NodeMerger): The strategy to use for finding and merging nodes
        """
        # Find groups of nodes to merge
        merge_candidates = merger.find_merge_candidates(self)

        # Process each group
        for group in merge_candidates:
            if len(group) < 2:
                continue

            # Generate name for merged node
            merged_node_name = merger.generate_merged_node_name(group, self)

            # Create the merged node if it doesn't exist yet
            if not self.has_node(merged_node_name):
                self.add_node(merged_node_name, sources=[])

            # Combine all sources and edges
            for node in group:
                if node == merged_node_name:
                    continue  # Skip if this is already the merged node

                # Combine sources
                node_attrs = self.get_node_attrs(node)
                if "sources" in node_attrs:
                    merged_attrs = self.get_node_attrs(merged_node_name)
                    merged_attrs["sources"].extend(node_attrs["sources"])
                    self.add_node(merged_node_name, **merged_attrs)

                # Redirect outgoing edges
                for _, target, data in self.get_out_edges(node):
                    if not self.has_edge(merged_node_name, target):
                        self.add_edge(
                            merged_node_name,
                            target,
                            relation=data.get("relation", ""),
                            sources=[],
                        )

                    # Combine the sources
                    if "sources" in data:
                        edges_data = self.get_edge_attrs(merged_node_name, target)
                        for edge_data in edges_data:
                            if isinstance(edge_data["sources"], str):
                                edge_data["sources"] = json.loads(edge_data["sources"])
                            if isinstance(data["sources"], str):
                                data["sources"] = json.loads(data["sources"])
                            edge_data["sources"].extend(data["sources"])
                            self.add_edge(merged_node_name, target, **edge_data)

                # Redirect incoming edges
                for source, _, data in self.get_in_edges(node):
                    if not self.has_edge(source, merged_node_name):
                        self.add_edge(
                            source,
                            merged_node_name,
                            relation=data.get("relation", ""),
                            sources=[],
                        )

                    # Combine the sources
                    if "sources" in data:
                        edges_data = self.get_edge_attrs(source, merged_node_name)
                        for edge_data in edges_data:
                            if isinstance(edge_data["sources"], str):
                                edge_data["sources"] = json.loads(edge_data["sources"])
                            if isinstance(data["sources"], str):
                                data["sources"] = json.loads(data["sources"])

                            edge_data["sources"].extend(data["sources"])
                            self.add_edge(source, merged_node_name, **edge_data)

                # Remove the original node
                self.remove_node(node)

    @property
    def triplets(self) -> List[Triplet]:
        """
        Get the list of triplets in the knowledge graph.

        Returns:
            List[Triplet]: The list of triplets
        """
        triplets = []
        for source, target, data in self.get_edges_data():
            relation = data.get("relation", "")

            # For each source in the edge's sources list, create a triplet
            if "sources" in data and data["sources"]:
                for source_data in data["sources"]:
                    # Extract metadata (excluding relation and triplet_id)
                    metadata = {
                        k: v
                        for k, v in source_data.items()
                        if k != "relation" and k != "triplet_id"
                    }
                    triplets.append(
                        Triplet(
                            node_1=source,
                            edge=relation,
                            node_2=target,
                            metadata=metadata,
                        )
                    )
            else:
                # If no sources, create a single triplet with no metadata
                triplets.append(Triplet(node_1=source, edge=relation, node_2=target))
        return triplets

    def __str__(self) -> str:
        """
        Get a string representation of the knowledge graph.

        Returns:
            str: A string representation
        """
        nodes_count = self.number_of_nodes()
        edge_types_count = len(self.get_edges())
        edges_count = self.number_of_edges()

        return f"KnowledgeGraph(nodes={nodes_count}, edge_types={edge_types_count}, edges={edges_count})"

    def save_to_file(self, filename: str) -> None:
        """
        Save the knowledge graph to a local file using JSON serialization.

        Args:
            filename (str): Path to the file where the graph will be saved

        Raises:
            ValueError: If the graph is empty
        """
        if not self.has_graph():
            raise ValueError("Cannot save an empty knowledge graph")

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)

        # Convert the graph to a serializable format
        serializable_graph = self._backend.to_serializable()
        data_to_save = {"graph": serializable_graph, "config": self.config}

        # Save using JSON
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(data_to_save, f, indent=2)

    @classmethod
    def load_from_file(
        cls, filename: str, scientific_domain: str, max_items: int = None
    ) -> "KnowledgeGraph":
        """
        Load a knowledge graph from a local file using JSON deserialization.
        If max_items is specified, a subgraph with that many triplets is returned.

        Args:
            filename (str): Path to the file containing the saved graph
            scientific_domain (str): The scientific domain for the graph.
            max_items (int, optional): The maximum number of triplets to load.

        Returns:
            KnowledgeGraph: A new KnowledgeGraph instance loaded from the file.

        Raises:
            FileNotFoundError: If the specified file doesn't exist
            ValueError: If the file format is invalid
        """
        if not os.path.exists(filename):
            raise FileNotFoundError(f"File not found: {filename}")

        # Load the data from file
        try:
            with open(filename, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            raise ValueError(f"Error loading file {filename}: {str(e)}")

        if not isinstance(data, dict) or "graph" not in data:
            raise ValueError(f"Invalid knowledge graph file format: {filename}")

        # Create a new knowledge graph instance from the full file
        full_kg = cls(
            scientific_domain=scientific_domain, config=data.get("config", {})
        )
        full_kg._backend = NetworkXBackend.from_serializable(
            data["graph"], scientific_domain
        )

        if max_items is None:
            return full_kg

        # If max_items is specified, create and return a subgraph
        logger.info(
            f"Note: 'max_items' is set to {max_items}. "
            "Returning a subgraph of the loaded file."
        )
        subgraph = cls(scientific_domain=scientific_domain, backend="networkx")

        # Get triplets and limit them
        triplets_to_load = full_kg.triplets[:max_items]

        subgraph.add_triplets(triplets_to_load)

        return subgraph

    def random_walk(self, start_node: str, max_steps: int = 10) -> List[str]:
        """
        Perform a random walk on the graph starting from a given node.

        Args:
            start_node (str): The node to start the walk from
            max_steps (int): The maximum number of steps to take

        Returns:
            List[str]: A list of nodes visited during the walk
        """
        if not self.has_node(start_node):
            raise ValueError(f"Start node '{start_node}' not found in the graph")

        current_node = start_node
        visited = [current_node]

        for _ in range(max_steps):
            neighbors = self.get_successors(current_node) + self.get_predecessors(
                current_node
            )
            if not neighbors:
                break
            next_node = random.choice(neighbors)
            visited.append(next_node)
            current_node = next_node

        return visited

    @classmethod
    def load_from_neo4j(
        cls, neo4j_config: dict, scientific_domain: str
    ) -> "KnowledgeGraph":
        """
        Load a knowledge graph from a Neo4j database.

        Args:
            neo4j_config (dict): Configuration dictionary for Neo4j connection with keys:
                - uri (str): URI for the Neo4j database connection (e.g., 'neo4j+s://hostname:port')
                - user (str): Neo4j database username
                - password (str): Neo4j database password
                - database (str, optional): Neo4j database name (for multi-database setups)
                - Additional configuration options can be included as needed

        Returns:
            KnowledgeGraph: A new KnowledgeGraph instance connected to the Neo4j database

        Raises:
            ConnectionError: If connection to Neo4j fails
            ValueError: If required configuration parameters are missing
        """
        # Validate required configuration parameters
        required_params = ["uri", "user", "password"]
        for param in required_params:
            if param not in neo4j_config:
                raise ValueError(
                    f"Missing required Neo4j configuration parameter: {param}"
                )

        try:
            # Create a knowledge graph with Neo4j backend
            kg = cls(
                backend="neo4j",
                uri=neo4j_config["uri"],
                user=neo4j_config["user"],
                password=neo4j_config["password"],
                scientific_domain=scientific_domain,
            )

            # Validate connection by ensuring we can access the database
            test_node = kg.get_random_node()

            node_count = len(kg.get_nodes()) if hasattr(kg, "get_nodes") else 0

            if test_node is None and node_count == 0:
                logger.warning(
                    "Connected to Neo4j, but no nodes were found that match the criteria"
                )

            return kg

        except Exception as e:
            raise ConnectionError(f"Failed to connect to Neo4j database: {str(e)}")
