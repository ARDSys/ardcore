import ast
import csv
import json
import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
)

import networkx as nx
import pandas as pd

if TYPE_CHECKING:
    from ard.data.dataset_item import DatasetItem

T = TypeVar("T", bound="DatasetItem")


class TripletMergeStrategy(Enum):
    """Enum for different strategies to merge metadata from multiple Triplets."""

    CONCAT = "concat"


@dataclass
class Triplet:
    """
    A single triplet (node_1, edge, node_2) representing a knowledge graph edge.

    Attributes:
        node_1 (str): The source node (subject)
        edge (str): The edge type (predicate/relation)
        node_2 (str): The target node (object)
        metadata (Dict[str, Any]): Additional metadata about the triplet, such as chunk_id, confidence, etc.
    """

    node_1: str
    edge: str
    node_2: str
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        """Initialize metadata as an empty dict if None."""
        if self.metadata is None:
            self.metadata = {}

    @property
    def chunk_id(self) -> Optional[str]:
        """
        Get the chunk_id from metadata for backward compatibility.

        Returns:
            Optional[str]: The chunk_id if it exists in metadata
        """
        return self.metadata.get("chunk_id")

    @chunk_id.setter
    def chunk_id(self, value: Optional[str]):
        """
        Set the chunk_id in metadata for backward compatibility.

        Args:
            value (Optional[str]): The chunk_id to set
        """
        if value is not None:
            self.metadata["chunk_id"] = value

    def __str__(self) -> str:
        return f"({self.node_1}, {self.edge}, {self.node_2})"

    def to_dict(self) -> Dict[str, Any]:
        """Convert the triplet to a dictionary format."""
        result = {
            "node_1": self.node_1,
            "edge": self.edge,
            "node_2": self.node_2,
        }

        # Add metadata fields
        if self.metadata:
            result["metadata"] = {}
            for key, value in self.metadata.items():
                result["metadata"][key] = value if value is not None else ""

        return result

    def merge_metadata(
        self, other, strategy: TripletMergeStrategy = TripletMergeStrategy.CONCAT
    ):
        """
        Merge metadata from another Triplet instance.

        Args:
            other (Triplet): The other Triplet instance to merge metadata from
            strategy (TripletMergeStrategy): The strategy to merge metadata.

        Returns:
            Triplet: A new Triplet instance with merged metadata
        """
        if strategy == TripletMergeStrategy.CONCAT:
            self.metadata = {**self.metadata, **other.metadata}
        else:
            raise ValueError(f"Invalid strategy: {strategy}")


class Triplets:
    """
    A collection of triplets representing a knowledge graph for a dataset item.

    This class manages triplets extracted from a dataset item, along with
    metadata about the extraction process and the graph structure.

    Attributes:
        triplets (List[Triplet]): List of triplets in the graph
        config (Dict): Configuration/metadata about the graph generation
        item_metadata (Dict): Metadata about the source dataset item
        graph (Optional[nx.DiGraph]): NetworkX directed graph representation of the triplets,
                                     built on-demand if not initialized with build_graph=True
    """

    def __init__(
        self,
        triplets: List[Triplet],
        config: Dict,
        item_metadata: Dict,
        build_graph: bool = False,
    ) -> None:
        """
        Initialize a Triplets object with triplets, config, and item metadata.

        Args:
            triplets (List[Triplet]): List of triplets in the graph
            config (Dict): Configuration/metadata about the graph generation
            item_metadata (Dict): Metadata about the source dataset item
            build_graph (bool): Whether to build the graph during initialization.
                               If False, the graph will be built on-demand when needed.
        """
        self.triplets = triplets
        self.config = config
        self.item_metadata = item_metadata
        self._graph = None

        if build_graph:
            self._build_graph()

    @classmethod
    def from_csv(
        cls,
        csv_path: Union[str, Path],
        config_path: Union[str, Path],
        metadata_path: Union[str, Path],
        build_graph: bool = False,
    ) -> "Triplets":
        """
        Create a Triplets object from CSV file of triplets and JSON config/metadata files.

        Args:
            csv_path (Union[str, Path]): Path to the CSV file containing triplets
            config_path (Union[str, Path]): Path to the JSON file containing graph config
            metadata_path (Union[str, Path]): Path to the JSON file containing item metadata
            build_graph (bool): Whether to build the graph during initialization

        Returns:
            Triplets: A new Triplets instance

        Raises:
            FileNotFoundError: If any of the required files don't exist
            ValueError: If the CSV file doesn't have the expected columns
        """
        # Load triplets from CSV
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Triplets file not found: {csv_path}")

        triplets = []
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if not all(
                col in reader.fieldnames for col in ["node_1", "node_2", "edge"]
            ):
                raise ValueError(
                    f"CSV file {csv_path} must have 'node_1', 'node_2', and 'edge' columns"
                )

            for row in reader:
                triplets.append(
                    Triplet(
                        node_1=row["node_1"],
                        edge=row["edge"],
                        node_2=row["node_2"],
                        metadata=row,
                    )
                )

        # Load config from JSON
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        # Load metadata from JSON
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        return cls(triplets, config, metadata, build_graph=build_graph)

    @classmethod
    def from_dataset_item(
        cls,
        item: Type[T],
        kg_version: str,
        build_graph: bool = False,
    ) -> "Triplets":
        """
        Create a Triplets object from a DatasetItem's storage.

        Args:
            item (DatasetItem): The DatasetItem to load the triplets from
            kg_version (str): The knowledge graph version to load (e.g., 'baseline_1')
            build_graph (bool): Whether to build the graph during initialization

        Returns:
            Triplets: A new Triplets instance

        Raises:
            FileNotFoundError: If the required files don't exist in the DatasetItem's storage
        """
        from ard.data.dataset_item import DataCategory

        # Load the DatasetItem
        item = item.from_local(item.id, item._storage_backend_name)

        # Get paths to the required files
        kg_dir = f"{DataCategory.KG.value}/{kg_version}"

        # Load triplets CSV
        triplets_data = item.get_file("triplets.csv", kg_dir)
        triplets = []

        # Parse CSV from bytes
        lines = triplets_data.decode("utf-8").splitlines()
        reader = csv.DictReader(lines)

        if not all(col in reader.fieldnames for col in ["node_1", "node_2", "edge"]):
            raise ValueError(
                "CSV data must have 'node_1', 'node_2', and 'edge' columns"
            )

        for row in reader:
            metadata_value = row.get("metadata", "{}")

            # Handle metadata parsing
            if isinstance(metadata_value, dict):
                # Already a dict, use directly
                metadata = metadata_value
            else:
                # It's a string, try to parse it
                try:
                    # Try JSON first
                    metadata = json.loads(metadata_value)
                except json.JSONDecodeError:
                    try:
                        # If not valid JSON, try parsing as Python literal
                        if metadata_value and metadata_value.strip():
                            metadata = ast.literal_eval(metadata_value)
                        else:
                            metadata = {}
                    except (SyntaxError, ValueError):
                        # If all parsing fails, use an empty dict
                        metadata = {}

            triplets.append(
                Triplet(
                    node_1=row["node_1"],
                    edge=row["edge"],
                    node_2=row["node_2"],
                    metadata=metadata,
                )
            )

        # Load config JSON
        config_data = item.get_file("config.json", kg_dir)
        config = json.loads(config_data.decode("utf-8"))

        # Get item metadata
        metadata = item.get_metadata().to_dict()

        return cls(triplets, config, metadata, build_graph=build_graph)

    def _build_graph(self) -> nx.DiGraph:
        """
        Build a NetworkX directed graph from the triplets.

        Returns:
            nx.DiGraph: A directed graph representation of the triplets
        """
        G = nx.DiGraph()

        # Add nodes and edges
        for triplet in self.triplets:
            if not G.has_node(triplet.node_1):
                G.add_node(triplet.node_1)
            if not G.has_node(triplet.node_2):
                G.add_node(triplet.node_2)

            G.add_edge(
                triplet.node_1,
                triplet.node_2,
                relation=triplet.edge,
                chunk_id=triplet.chunk_id,
            )

        self._graph = G
        return G

    @property
    def graph(self) -> nx.DiGraph:
        """
        Get the graph representation of the triplets, building it if it doesn't exist yet.

        Returns:
            nx.DiGraph: A directed graph representation of the triplets
        """
        if self._graph is None:
            self._build_graph()
        return self._graph

    def has_graph(self) -> bool:
        """
        Check if the graph has been built.

        Returns:
            bool: True if the graph has been built, False otherwise
        """
        return self._graph is not None

    def to_csv(self, output_path: Union[str, Path]) -> None:
        """
        Save the triplets to a CSV file.

        Args:
            output_path (Union[str, Path]): Path where the CSV file should be saved
        """
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f, fieldnames=["node_1", "node_2", "edge", "metadata"]
            )
            writer.writeheader()
            for triplet in self.triplets:
                writer.writerow(triplet.to_dict())

    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert the triplets to a pandas DataFrame.

        Returns:
            pd.DataFrame: DataFrame containing the triplets
        """
        return pd.DataFrame([t.to_dict() for t in self.triplets])

    def save_to_dataset_item(
        self,
        item_id: str,
        kg_version: str,
        daset_item_type: Type["DatasetItem"],
        storage_backend: Optional[str] = None,
    ) -> None:
        """
        Save the triplets and config to a DatasetItem's storage.

        Args:
            item_id (str): The ID of the DatasetItem
            kg_version (str): The knowledge graph version to save as (e.g., 'baseline_1')
            storage_backend (Optional[str]): Optional storage backend name
        """
        from ard.data.dataset_item import DataCategory

        # Load the DatasetItem
        item = daset_item_type.from_local(item_id, storage_backend)

        # Create KG version directory if it doesn't exist
        kg_dir = f"{DataCategory.KG.value}/{kg_version}"
        # Save triplets as CSV
        csv_data = self.to_dataframe().to_csv(index=False).encode("utf-8")
        item.save_file("triplets.csv", csv_data, kg_dir)

        # Save config as JSON
        config_data = json.dumps(self.config, indent=2).encode("utf-8")
        item.save_file("config.json", config_data, kg_dir)

        # # Save graph as pickle
        # graph_data = nx.to_pickle(self.graph)
        # item.save_file("graph.gpickle", graph_data, kg_dir)

    def get_nodes(self) -> Set[str]:
        """
        Get all unique nodes in the graph.

        If the graph has not been built, this method will return a set of unique nodes
        extracted directly from the triplets without building the graph.

        Returns:
            Set[str]: Set of all node names
        """
        if not self.has_graph():
            import warnings

            warnings.warn(
                "Graph not initialized. Extracting nodes directly from triplets without building the graph."
            )
            nodes = set()
            for triplet in self.triplets:
                nodes.add(triplet.node_1)
                nodes.add(triplet.node_2)
            return nodes

        return set(self.graph.nodes())

    def get_edges(self) -> Set[str]:
        """
        Get all unique edge types in the graph.

        If the graph has not been built, this method will return a set of unique edge types
        extracted directly from the triplets without building the graph.

        Returns:
            Set[str]: Set of all edge types
        """
        if not self.has_graph():
            import warnings

            warnings.warn(
                "Graph not initialized. Extracting edge types directly from triplets without building the graph."
            )
            return {triplet.edge for triplet in self.triplets}

        return {data["relation"] for _, _, data in self.graph.edges(data=True)}

    def get_node_neighbors(self, node: str) -> List[Tuple[str, str, str]]:
        """
        Get all neighbors of a node along with the edge types.

        If the graph has not been built, this method will extract neighbors directly from the triplets
        without building the graph.

        Args:
            node (str): The node to get neighbors for

        Returns:
            List[Tuple[str, str, str]]: List of (source, relation, target) tuples
        """
        if not self.has_graph():
            import warnings

            warnings.warn(
                "Graph not initialized. Extracting neighbors directly from triplets without building the graph."
            )
            neighbors = []
            for triplet in self.triplets:
                if triplet.node_1 == node:
                    neighbors.append((node, triplet.edge, triplet.node_2))
                elif triplet.node_2 == node:
                    neighbors.append((triplet.node_1, triplet.edge, node))
            return neighbors

        if node not in self.graph:
            return []

        neighbors = []

        # Outgoing edges
        for _, target, data in self.graph.out_edges(node, data=True):
            neighbors.append((node, data["relation"], target))

        # Incoming edges
        for source, _, data in self.graph.in_edges(node, data=True):
            neighbors.append((source, data["relation"], node))

        return neighbors

    def get_subgraph(self, nodes: List[str]) -> "Triplets":
        """
        Extract a subgraph containing only the specified nodes and their connections.

        If the graph has not been built, this method will extract the subgraph directly from the triplets
        without building the graph.

        Args:
            nodes (List[str]): List of nodes to include in the subgraph

        Returns:
            Triplets: A new Triplets instance containing only the specified nodes
        """
        if not self.has_graph():
            import warnings

            warnings.warn(
                "Graph not initialized. Extracting subgraph directly from triplets without building the graph."
            )
            nodes_set = set(nodes)
            subgraph_triplets = []
            for triplet in self.triplets:
                if triplet.node_1 in nodes_set and triplet.node_2 in nodes_set:
                    subgraph_triplets.append(triplet)
            return Triplets(subgraph_triplets, self.config, self.item_metadata)

        # Create subgraph with only the specified nodes
        subgraph = self.graph.subgraph(nodes)

        # Extract triplets from the subgraph
        subgraph_triplets = []
        for source, target, data in subgraph.edges(data=True):
            subgraph_triplets.append(
                Triplet(
                    node_1=source,
                    edge=data["relation"],
                    node_2=target,
                    metadata=data,
                )
            )

        # Create new Triplets object with the same config and metadata
        return Triplets(subgraph_triplets, self.config, self.item_metadata)

    def __len__(self) -> int:
        """
        Get the number of triplets in the graph.

        Returns:
            int: Number of triplets
        """
        return len(self.triplets)

    def __str__(self) -> str:
        """
        Get a string representation of the Triplets object.

        Returns:
            str: String representation
        """
        return f"Triplets(triplets={len(self.triplets)}, nodes={len(self.get_nodes())}, edges={len(self.get_edges())})"
