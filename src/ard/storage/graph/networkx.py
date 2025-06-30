import random
from typing import Any, Dict, List, Set, Tuple

import networkx as nx

from ard.storage.graph import GraphBackend


class NetworkXBackend(GraphBackend):
    """
    NetworkX backend implementation for the knowledge graph.
    """

    def __init__(self):
        """Initialize a new NetworkX backend."""
        self._graph = nx.MultiDiGraph()

    @classmethod
    def from_networkx(cls, graph: nx.MultiDiGraph):
        """Initialize a new NetworkX backend from a NetworkX graph."""
        backend = cls()
        backend._graph = graph
        return backend

    @classmethod
    def from_serializable(cls, data: Dict[str, Any], scientific_domain: str):
        """
        Initialize a new NetworkX backend from a serialized dictionary.

        Args:
            data (Dict[str, Any]): Serialized graph data from to_serializable()

        Returns:
            NetworkXBackend: New backend instance
        """
        backend = cls()

        # Add all nodes with their attributes
        for node_data in data["nodes"]:
            node_id = node_data["id"]
            attrs = node_data.get("attributes", {})
            # Remove scientific_domain from attrs to avoid passing scientific_domain two times
            attrs.pop("scientific_domain", None)
            backend.add_node(node_id, scientific_domain, **attrs)

        # Add all edges with their attributes
        for edge_data in data["edges"]:
            source = edge_data["source"]
            target = edge_data["target"]
            attrs = edge_data.get("attributes", {})
            # Remove scientific_domain from attrs to avoid passing scientific_domain two times
            attrs.pop("scientific_domain", None)
            backend.add_edge(
                source, target, scientific_domain=scientific_domain, **attrs
            )

        return backend

    def to_serializable(self) -> Dict[str, Any]:
        """
        Convert the NetworkX graph to a serializable dictionary.

        Returns:
            Dict[str, Any]: A serializable representation of the graph
        """
        # Create serializable structures for nodes and edges
        nodes = []
        for node, attrs in self._graph.nodes(data=True):
            nodes.append({"id": node, "attributes": attrs})

        edges = []
        for source, target, attrs in self._graph.edges(data=True):
            edges.append({"source": source, "target": target, "attributes": attrs})

        return {"nodes": nodes, "edges": edges}

    def has_node(self, node: str, scientific_domain: str) -> bool:
        """
        Check if a node exists in a specific scientific domain.

        Args:
            node: The node identifier to check
            scientific_domain: The scientific domain to check for

        Returns:
            bool: True if the node exists and belongs to the specified scientific domain
        """
        # Check if node exists at all
        if not self._graph.has_node(node):
            return False

        # Check if node has the specific scientific domain
        node_attrs = self.get_node_attrs(node, scientific_domain)
        domains = node_attrs.get("scientific_domain", [])

        # Return True if the scientific domain is in the node's domains
        return scientific_domain in domains

    def add_node(self, node: str, scientific_domain: str, **attrs) -> None:
        """
        Create a node if it doesn't exist, or update an existing node.

        This function will:
        - Create a new node with the given scientific_domain and attributes if the node doesn't exist
        - Add the specified scientific_domain to the list of domains for an existing node
        - Update node properties in both cases

        Args:
            node: The name/identifier of the node
            scientific_domain: The scientific domain to ensure the node has
            **attrs: Optional attributes to set on the node
        """
        # Check if node exists at all (regardless of scientific domain)
        if self._graph.has_node(node):
            # Get existing node attributes
            node_attrs = dict(self._graph.nodes[node])

            # Add the scientific domain to the list if it doesn't exist
            if "scientific_domain" not in node_attrs:
                node_attrs["scientific_domain"] = [scientific_domain]
            elif scientific_domain not in node_attrs["scientific_domain"]:
                node_attrs["scientific_domain"].append(scientific_domain)

            # Update with new attributes
            node_attrs.update(attrs)

            # Update the node
            self._graph.add_node(node, **node_attrs)
        else:
            # Create a new node with scientific_domain as a list
            attrs["scientific_domain"] = [scientific_domain]
            self._graph.add_node(node, **attrs)

    def add_edge(
        self, source: str, target: str, relation: str, scientific_domain: str, **attrs
    ) -> None:
        """
        Create an edge between source and target nodes.

        This function will:
        - Create a new edge with the given attributes if no edge with the specified relation exists
        - Update the edge's scientific_domain and attributes if an edge with the specified relation already exists

        Args:
            source: The source node identifier
            target: The target node identifier
            relation: The relationship type
            scientific_domain: The scientific domain for the edge
            **attrs: Optional attributes to set on the edge
        """
        # Make a copy of attributes to avoid modifying the input
        edge_attrs = attrs.copy()

        # Add relation to attributes
        edge_attrs["relation"] = relation

        # Check if an edge with this relation already exists
        existing_edge_key = None
        if self._graph.has_edge(source, target):
            for key, data in self._graph.get_edge_data(source, target).items():
                if data.get("relation") == relation:
                    existing_edge_key = key
                    break

        if existing_edge_key is not None:
            # Get existing edge attributes
            existing_attrs = self._graph.get_edge_data(source, target)[
                existing_edge_key
            ]

            # Update scientific_domain list
            if "scientific_domain" not in existing_attrs:
                existing_attrs["scientific_domain"] = [scientific_domain]
            elif scientific_domain not in existing_attrs["scientific_domain"]:
                existing_attrs["scientific_domain"].append(scientific_domain)

            # Update with new attributes
            existing_attrs.update(edge_attrs)

            # Remove the existing edge
            self._graph.remove_edge(source, target, existing_edge_key)

            # Add it back with updated attributes
            self._graph.add_edge(source, target, **existing_attrs)
        else:
            # Create new edge with scientific_domain as a list
            edge_attrs["scientific_domain"] = [scientific_domain]

            # Add the new edge
            self._graph.add_edge(source, target, **edge_attrs)

    def has_edge(self, source: str, target: str, scientific_domain: str) -> bool:
        """
        Check if an edge exists between source and target in the specified scientific domain.

        Args:
            source: The source node identifier
            target: The target node identifier
            scientific_domain: The scientific domain to check for

        Returns:
            bool: True if the edge exists and belongs to the specified scientific domain
        """
        # Check if the edge exists at all
        if not self._graph.has_edge(source, target):
            return False

        # Get edge attributes
        edge_data = self._graph.get_edge_data(source, target)

        # Check all parallel edges (for MultiDiGraph)
        for key in edge_data:
            attrs = edge_data[key]
            domains = attrs.get("scientific_domain", [])
            if scientific_domain in domains:
                return True

        return False

    def get_node_attrs(self, node: str, scientific_domain: str) -> Dict[str, Any]:
        """
        Get attributes of a node in the specified scientific domain.

        Args:
            node: The node identifier
            scientific_domain: The scientific domain to check for

        Returns:
            Dict[str, Any]: Node attributes if the node belongs to the domain, otherwise empty dict
        """
        # Check if node exists
        if not self._graph.has_node(node):
            return {}

        # Get node attributes
        attrs = dict(self._graph.nodes[node])

        # Check if node belongs to the specified domain
        domains = attrs.get("scientific_domain", [])
        if scientific_domain not in domains:
            return {}

        return attrs

    def get_edge_attrs(
        self, source: str, target: str, scientific_domain: str
    ) -> List[Dict[str, Any]]:
        """
        Get attributes of all edges between source and target in the specified scientific domain.

        Args:
            source: The source node identifier
            target: The target node identifier
            scientific_domain: The scientific domain to filter by

        Returns:
            List[Dict[str, Any]]: List of attribute dictionaries for all matching edges
        """
        # Check if any edge exists between the nodes
        if not self._graph.has_edge(source, target):
            return []

        # Get all edges between source and target
        edge_data = self._graph.get_edge_data(source, target)

        # Collect attributes for all edges in the specified scientific domain
        result = []
        for key, attrs in edge_data.items():
            domains = attrs.get("scientific_domain", [])
            if scientific_domain in domains:
                result.append(dict(attrs))

        return result

    def get_nodes(self, scientific_domain: str) -> Set[str]:
        """
        Get all nodes belonging to the specified scientific domain.

        Args:
            scientific_domain: The scientific domain to filter by

        Returns:
            Set[str]: Set of node identifiers belonging to the specified domain
        """
        # Filter nodes by scientific domain
        filtered_nodes = set()
        for node, attrs in self._graph.nodes(data=True):
            domains = attrs.get("scientific_domain", [])
            if scientific_domain in domains:
                filtered_nodes.add(node)

        return filtered_nodes

    def get_edges(
        self, scientific_domain: str
    ) -> List[Tuple[str, str, Dict[str, Any]]]:
        """
        Get all edges belonging to the specified scientific domain.

        Args:
            scientific_domain: The scientific domain to filter by

        Returns:
            List[Tuple[str, str, Dict[str, Any]]]: List of (source, target, attrs) tuples
            for edges belonging to the specified domain
        """
        # Filter edges by scientific domain
        filtered_edges = []
        for source, target, attrs in self._graph.edges(data=True):
            domains = attrs.get("scientific_domain", [])
            if scientific_domain in domains:
                filtered_edges.append((source, target, attrs))

        return filtered_edges

    def get_successors(
        self, node: str, scientific_domain: str, n_of_nodes: int = None
    ) -> List[str]:
        """
        Get all successor nodes of a node that belong to the specified scientific domain.

        Args:
            node: The node identifier
            scientific_domain: The scientific domain to filter by
            n_of_nodes: The number of successors to return
        Returns:
            List[str]: List of successor node identifiers that belong to the specified domain
        """
        # Get all basic successors
        all_successors = self._graph.successors(node)

        # Filter successors by scientific domain
        domain_successors = []
        for successor in all_successors:
            attrs = dict(self._graph.nodes[successor])
            domains = attrs.get("scientific_domain", [])
            if scientific_domain in domains:
                domain_successors.append(successor)

        if n_of_nodes is None:
            return domain_successors
        else:
            random.shuffle(domain_successors)
            return domain_successors[:n_of_nodes]

    def get_predecessors(
        self, node: str, scientific_domain: str, n_of_nodes: int = None
    ) -> List[str]:
        """
        Get all predecessor nodes of a node that belong to the specified scientific domain.

        Args:
            node: The node identifier
            scientific_domain: The scientific domain to filter by

        Returns:
            List[str]: List of predecessor node identifiers that belong to the specified domain
        """
        # Get all basic predecessors
        all_predecessors = self._graph.predecessors(node)

        # Filter predecessors by scientific domain
        domain_predecessors = []
        for predecessor in all_predecessors:
            attrs = dict(self._graph.nodes[predecessor])
            domains = attrs.get("scientific_domain", [])
            if scientific_domain in domains:
                domain_predecessors.append(predecessor)

        if n_of_nodes is None:
            return domain_predecessors
        else:
            random.shuffle(domain_predecessors)
            return domain_predecessors[:n_of_nodes]

    def get_out_edges(
        self, node: str, scientific_domain: str
    ) -> List[Tuple[str, str, Dict[str, Any]]]:
        """
        Get all outgoing edges of a node that belong to the specified scientific domain.

        Args:
            node: The node identifier
            scientific_domain: The scientific domain to filter by

        Returns:
            List[Tuple[str, str, Dict[str, Any]]]: List of (source, target, attrs) tuples
            for outgoing edges that belong to the specified domain
        """
        # Get all outgoing edges
        all_edges = self._graph.out_edges(node, data=True)

        # Filter edges by scientific domain
        domain_edges = []
        for source, target, attrs in all_edges:
            domains = attrs.get("scientific_domain", [])
            if scientific_domain in domains:
                domain_edges.append((source, target, attrs))

        return domain_edges

    def get_in_edges(
        self, node: str, scientific_domain: str
    ) -> List[Tuple[str, str, Dict[str, Any]]]:
        """
        Get all incoming edges of a node that belong to the specified scientific domain.

        Args:
            node: The node identifier
            scientific_domain: The scientific domain to filter by

        Returns:
            List[Tuple[str, str, Dict[str, Any]]]: List of (source, target, attrs) tuples
            for incoming edges that belong to the specified domain
        """
        # Get all incoming edges
        all_edges = self._graph.in_edges(node, data=True)

        # Filter edges by scientific domain
        domain_edges = []
        for source, target, attrs in all_edges:
            domains = attrs.get("scientific_domain", [])
            if scientific_domain in domains:
                domain_edges.append((source, target, attrs))

        return domain_edges

    def remove_node(self, node: str, scientific_domain: str) -> None:
        """
        Remove a node from the specified scientific domain.

        This function will:
        - Remove the node entirely if it only belongs to the specified scientific domain
        - Otherwise, just remove the specified scientific domain from the node's domain list

        Args:
            node: The node identifier
            scientific_domain: The scientific domain to remove the node from
        """
        # Check if node exists
        if not self._graph.has_node(node):
            return

        # Get the node's attributes
        node_attrs = dict(self._graph.nodes[node])
        domains = node_attrs.get("scientific_domain", [])

        # If node doesn't belong to this domain, nothing to do
        if scientific_domain not in domains:
            return

        # Remove this domain from the list
        domains.remove(scientific_domain)

        if not domains:
            # If no domains left, remove the node entirely
            self._graph.remove_node(node)
        else:
            # Update the node with the modified domains list
            node_attrs["scientific_domain"] = domains

            # Create a new node with the updated attributes
            # First, store a copy of all attributes
            attrs_copy = node_attrs.copy()

            # Remove the node
            self._graph.remove_node(node)

            # Add it back with updated attributes
            self._graph.add_node(node, **attrs_copy)

    def number_of_edges(self, scientific_domain: str) -> int:
        """
        Get the total number of edges belonging to the specified scientific domain.

        Args:
            scientific_domain: The scientific domain to filter by

        Returns:
            int: Number of edges in the specified domain
        """
        # Count edges with the specific scientific domain
        count = 0
        for _, _, attrs in self._graph.edges(data=True):
            domains = attrs.get("scientific_domain", [])
            if scientific_domain in domains:
                count += 1

        return count

    def shortest_path(
        self, source: str, target: str, scientific_domain: str, directed: bool = True
    ) -> List[str]:
        """
        Get the shortest path between two nodes where edges belong to the specified scientific domain.

        Args:
            source: The source node identifier
            target: The target node identifier
            scientific_domain: The scientific domain to filter by
            directed: Whether to respect edge direction

        Returns:
            List[str]: List of node identifiers in the shortest path

        Raises:
            NetworkXNoPath: If no path exists in the specified domain
        """
        # Create a filtered graph containing only edges with the specified scientific domain
        filtered_graph = nx.MultiDiGraph() if directed else nx.MultiGraph()

        # Add all nodes
        for node in self._graph.nodes():
            filtered_graph.add_node(node)

        # Add only edges with the specified scientific domain
        for source_node, target_node, attrs in self._graph.edges(data=True):
            domains = attrs.get("scientific_domain", [])
            if scientific_domain in domains:
                filtered_graph.add_edge(source_node, target_node)

        # Convert to undirected if needed
        if not directed:
            filtered_graph = filtered_graph.to_undirected()

        # Find shortest path in the filtered graph
        return nx.shortest_path(filtered_graph, source, target)

    def __len__(self, scientific_domain: str) -> int:
        """
        Count the number of nodes belonging to the specified scientific domain.

        Args:
            scientific_domain: The scientific domain to filter by

        Returns:
            int: Number of nodes in the specified domain
        """
        # Count nodes with the specific scientific domain
        count = 0
        for _, attrs in self._graph.nodes(data=True):
            domains = attrs.get("scientific_domain", [])
            if scientific_domain in domains:
                count += 1

        return count

    @property
    def graph(self) -> nx.DiGraph:
        """Get the underlying NetworkX graph."""
        return self._graph
