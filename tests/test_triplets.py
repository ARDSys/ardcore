import json
import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from ardcore.data.triplets import Triplet, Triplets


@pytest.fixture
def sample_triplets():
    """Create a list of sample triplets for testing."""
    return [
        Triplet(
            node_1="microglia",
            edge="undergoes",
            node_2="activation",
            metadata={"chunk_id": "test_chunk_1"},
        ),
        Triplet(
            node_1="astrocyte",
            edge="undergoes",
            node_2="activation",
            metadata={"chunk_id": "test_chunk_1"},
        ),
        Triplet(
            node_1="gene expression analysis",
            edge="performed on",
            node_2="microglia",
            metadata={"chunk_id": "test_chunk_2"},
        ),
        Triplet(
            node_1="gene expression analysis",
            edge="performed on",
            node_2="astrocyte",
            metadata={"chunk_id": "test_chunk_2"},
        ),
        Triplet(
            node_1="microglia",
            edge="involved in",
            node_2="inflammatory response",
            metadata={"chunk_id": "test_chunk_3"},
        ),
    ]


@pytest.fixture
def sample_config():
    """Create a sample config dictionary for testing."""
    return {
        "chunk_method": "fixed",
        "chunk_size": 1000,
        "chunk_overlap": 100,
        "model_name": "gemini/gemini-2.0-flash",
        "timestamp": "20250304_120017",
    }


@pytest.fixture
def sample_metadata():
    """Create a sample metadata dictionary for testing."""
    return {
        "doi": "10.1186/s12974-020-01774-9",
        "pm_id": 32238175,
        "title": "Transcriptomic profiling of microglia and astrocytes throughout aging.",
        "abstract": "Sample abstract text for testing.",
        "id": "test_item_id",
    }


def test_triplet_creation():
    """Test that a Triplet can be created with the required attributes."""
    triplet = Triplet(
        node_1="microglia",
        edge="undergoes",
        node_2="activation",
        metadata={"chunk_id": "test_chunk_1"},
    )

    assert triplet.node_1 == "microglia"
    assert triplet.edge == "undergoes"
    assert triplet.node_2 == "activation"
    assert triplet.metadata["chunk_id"] == "test_chunk_1"
    assert str(triplet) == "(microglia, undergoes, activation)"

    # Test to_dict method
    triplet_dict = triplet.to_dict()
    assert triplet_dict["node_1"] == "microglia"
    assert triplet_dict["edge"] == "undergoes"
    assert triplet_dict["node_2"] == "activation"
    assert triplet_dict["metadata"]["chunk_id"] == "test_chunk_1"


def test_triplets_creation_with_graph(sample_triplets, sample_config, sample_metadata):
    """Test that a Triplets object can be created with triplets, config, and metadata with graph building."""
    triplets = Triplets(
        sample_triplets, sample_config, sample_metadata, build_graph=True
    )

    assert triplets.triplets == sample_triplets
    assert triplets.config == sample_config
    assert triplets.item_metadata == sample_metadata
    assert len(triplets) == 5

    # Test graph creation
    assert triplets._graph is not None
    assert len(triplets.graph.nodes()) == 5  # 5 unique nodes
    assert len(triplets.graph.edges()) == 5  # 5 edges

    # Test string representation
    assert str(triplets) == "Triplets(triplets=5, nodes=5, edges=3)"


def test_triplets_creation_without_graph(
    sample_triplets, sample_config, sample_metadata
):
    """Test that a Triplets object can be created without building the graph."""
    triplets = Triplets(
        sample_triplets, sample_config, sample_metadata, build_graph=False
    )

    assert triplets.triplets == sample_triplets
    assert triplets.config == sample_config
    assert triplets.item_metadata == sample_metadata
    assert len(triplets) == 5

    # Test that graph is not built yet
    assert triplets._graph is None

    # Test that accessing graph property builds the graph
    graph = triplets.graph
    assert triplets._graph is not None
    assert len(graph.nodes()) == 5
    assert len(graph.edges()) == 5

    # Test string representation
    assert str(triplets) == "Triplets(triplets=5, nodes=5, edges=3)"


def test_triplets_from_csv():
    """Test creating Triplets from CSV and JSON files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test files
        csv_path = os.path.join(temp_dir, "triplets.csv")
        config_path = os.path.join(temp_dir, "config.json")
        metadata_path = os.path.join(temp_dir, "metadata.json")

        # Write test CSV
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write("node_1,edge,node_2,chunk_id\n")
            f.write("microglia,undergoes,activation,test_chunk_1\n")
            f.write("astrocyte,undergoes,activation,test_chunk_1\n")

        # Write test config
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump({"chunk_method": "fixed", "chunk_size": 1000}, f)

        # Write test metadata
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump({"doi": "10.1186/test", "title": "Test Paper"}, f)

        # Create Triplets from files without building graph
        triplets = Triplets.from_csv(
            csv_path, config_path, metadata_path, build_graph=False
        )

        assert len(triplets) == 2
        assert triplets.config["chunk_method"] == "fixed"
        assert triplets.item_metadata["doi"] == "10.1186/test"
        assert triplets._graph is None

        # Create Triplets from files with building graph
        triplets_with_graph = Triplets.from_csv(
            csv_path, config_path, metadata_path, build_graph=True
        )
        assert triplets_with_graph._graph is not None

        # Test to_csv method
        output_path = os.path.join(temp_dir, "output.csv")
        triplets.to_csv(output_path)

        # Verify the output file
        with open(output_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            assert len(lines) == 3  # Header + 2 triplets
            assert "node_1,node_2,edge,metadata" in lines[0]


@patch("ardcore.data.dataset_item.DatasetItem")
def test_triplets_from_dataset_item(mock_dataset_item, sample_config, sample_metadata):
    """Test creating Triplets from a DatasetItem."""
    # Create mock DatasetItem
    mock_item = MagicMock()
    mock_dataset_item.from_local.return_value = mock_item

    # Mock the get_file method to return test data
    def mock_get_file(filename, category):
        if filename == "triplets.csv":
            return b"node_1,edge,node_2,chunk_id\nmicroglia,undergoes,activation,test_chunk_1"
        elif filename == "config.json":
            return json.dumps(sample_config).encode("utf-8")
        return b""

    mock_item.get_file.side_effect = mock_get_file

    # Mock the get_metadata method
    mock_metadata = MagicMock()
    mock_metadata.to_dict.return_value = sample_metadata
    mock_item.get_metadata.return_value = mock_metadata

    # Create Triplets from DatasetItem without building graph
    triplets = Triplets.from_dataset_item(
        mock_dataset_item, "baseline_1", build_graph=False
    )

    assert len(triplets) == 1
    assert triplets.triplets[0].node_1 == "microglia"
    assert triplets.config == sample_config
    assert triplets.item_metadata == sample_metadata
    assert triplets._graph is None

    # Create Triplets from DatasetItem with building graph
    triplets_with_graph = Triplets.from_dataset_item(
        mock_dataset_item, "baseline_1", build_graph=True
    )
    assert triplets_with_graph._graph is not None


def test_lazy_graph_building(sample_triplets, sample_config, sample_metadata):
    """Test that graph methods build the graph on-demand."""
    triplets = Triplets(
        sample_triplets, sample_config, sample_metadata, build_graph=False
    )

    # Initially, graph should not be built
    assert triplets._graph is None

    # Calling graph property should build the graph
    triplets.graph
    assert triplets._graph is not None

    # Reset for next test
    triplets = Triplets(
        sample_triplets, sample_config, sample_metadata, build_graph=False
    )
    assert triplets._graph is None

    # Verify has_graph method
    assert not triplets.has_graph()
    _ = triplets.graph
    assert triplets.has_graph()


def test_methods_without_graph(sample_triplets, sample_config, sample_metadata):
    """Test that methods work correctly without initializing the graph."""
    triplets = Triplets(
        sample_triplets, sample_config, sample_metadata, build_graph=False
    )

    # Initially, graph should not be built
    assert triplets._graph is None

    # Test get_nodes without graph
    with pytest.warns(UserWarning, match="Graph not initialized"):
        nodes = triplets.get_nodes()
    assert triplets._graph is None  # Graph should still not be built
    assert len(nodes) == 5
    assert "microglia" in nodes
    assert "activation" in nodes

    # Test get_edges without graph
    with pytest.warns(UserWarning, match="Graph not initialized"):
        edges = triplets.get_edges()
    assert triplets._graph is None  # Graph should still not be built
    assert len(edges) == 3
    assert "undergoes" in edges
    assert "performed on" in edges

    # Test get_node_neighbors without graph
    with pytest.warns(UserWarning, match="Graph not initialized"):
        neighbors = triplets.get_node_neighbors("microglia")
    assert triplets._graph is None  # Graph should still not be built
    assert len(neighbors) == 3
    assert ("microglia", "undergoes", "activation") in neighbors
    assert ("microglia", "involved in", "inflammatory response") in neighbors

    # Test get_subgraph without graph
    with pytest.warns(UserWarning, match="Graph not initialized"):
        subgraph = triplets.get_subgraph(["microglia", "activation"])
    assert triplets._graph is None  # Graph should still not be built
    assert len(subgraph) == 1
    assert subgraph.triplets[0].node_1 == "microglia"
    assert subgraph.triplets[0].edge == "undergoes"
    assert subgraph.triplets[0].node_2 == "activation"

    # Verify that the graph is still not built after all these operations
    assert not triplets.has_graph()


def test_get_nodes_and_edges(sample_triplets, sample_config, sample_metadata):
    """Test getting nodes and edges from Triplets."""
    triplets = Triplets(
        sample_triplets, sample_config, sample_metadata, build_graph=True
    )

    nodes = triplets.get_nodes()
    edges = triplets.get_edges()

    assert len(nodes) == 5
    assert "microglia" in nodes
    assert "activation" in nodes
    assert "inflammatory response" in nodes

    assert len(edges) == 3
    assert "undergoes" in edges
    assert "performed on" in edges
    assert "involved in" in edges


def test_get_node_neighbors(sample_triplets, sample_config, sample_metadata):
    """Test getting neighbors of a node."""
    triplets = Triplets(sample_triplets, sample_config, sample_metadata)

    # Test neighbors of microglia
    with pytest.warns(UserWarning, match="Graph not initialized"):
        neighbors = triplets.get_node_neighbors("microglia")
    assert len(neighbors) == 3
    assert ("microglia", "undergoes", "activation") in neighbors
    assert ("microglia", "involved in", "inflammatory response") in neighbors
    assert ("gene expression analysis", "performed on", "microglia") in neighbors

    # Test neighbors of gene expression analysis
    with pytest.warns(UserWarning, match="Graph not initialized"):
        neighbors = triplets.get_node_neighbors("gene expression analysis")
    assert len(neighbors) == 2
    assert ("gene expression analysis", "performed on", "microglia") in neighbors
    assert ("gene expression analysis", "performed on", "astrocyte") in neighbors

    # Test non-existent node
    with pytest.warns(UserWarning, match="Graph not initialized"):
        neighbors = triplets.get_node_neighbors("non_existent_node")
    assert len(neighbors) == 0


def test_get_subgraph(sample_triplets, sample_config, sample_metadata):
    """Test extracting a subgraph from Triplets."""
    triplets = Triplets(sample_triplets, sample_config, sample_metadata)

    # Extract subgraph with only microglia and activation
    with pytest.warns(UserWarning, match="Graph not initialized"):
        subgraph = triplets.get_subgraph(["microglia", "activation"])

    assert len(subgraph) == 1
    assert subgraph.triplets[0].node_1 == "microglia"
    assert subgraph.triplets[0].edge == "undergoes"
    assert subgraph.triplets[0].node_2 == "activation"

    # Extract larger subgraph
    with pytest.warns(UserWarning, match="Graph not initialized"):
        subgraph = triplets.get_subgraph(
            ["microglia", "activation", "inflammatory response"]
        )
    assert len(subgraph) == 2
