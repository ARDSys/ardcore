# from ard.knowledge_graph.knowledge_graph import KnowledgeGraph


# def test_knowledge_graph_creation():
#     """Test that a KnowledgeGraph can be created."""
#     kg = KnowledgeGraph()

#     assert kg.graph == {}
#     assert len(kg.get_nodes()) == 0
#     assert len(kg.get_edges()) == 0


# def test_add_node():
#     """Test adding nodes to the graph."""
#     kg = KnowledgeGraph()

#     kg.add_node("node1")
#     kg.add_node("node2")
#     kg.add_node("node3")

#     assert len(kg.get_nodes()) == 3
#     assert "node1" in kg.get_nodes()
#     assert "node2" in kg.get_nodes()
#     assert "node3" in kg.get_nodes()

#     # Test adding a duplicate node (should not increase count)
#     kg.add_node("node1")
#     assert len(kg.get_nodes()) == 3


# def test_add_edge():
#     """Test adding edges to the graph."""
#     kg = KnowledgeGraph()

#     kg.add_edge("source1", "target1", "relates_to")
#     kg.add_edge("source1", "target2", "cites")
#     kg.add_edge("source2", "target1", "contradicts")

#     # Check that nodes were automatically created
#     assert len(kg.get_nodes()) == 4
#     assert "source1" in kg.get_nodes()
#     assert "target1" in kg.get_nodes()
#     assert "target2" in kg.get_nodes()
#     assert "source2" in kg.get_nodes()

#     # Check edges
#     edges = kg.get_edges()
#     assert len(edges) == 3
#     assert ("source1", "target1", "relates_to") in edges
#     assert ("source1", "target2", "cites") in edges
#     assert ("source2", "target1", "contradicts") in edges

import os
import tempfile

import pytest

from ard.data.triplets import Triplet
from ard.knowledge_graph import KnowledgeGraph


@pytest.fixture
def sample_triplets_with_duplicates():
    """Create a sample list of triplets with duplicates for testing."""
    return [
        Triplet(
            node_1="microglia",
            edge="undergoes",
            node_2="activation",
            metadata={"chunk_id": "chunk_1", "confidence": 0.9},
        ),
        Triplet(
            node_1="microglia",
            edge="undergoes",
            node_2="activation",
            metadata={"chunk_id": "chunk_2", "source": "paper_1"},
        ),
        Triplet(
            node_1="tau",
            edge="accumulates",
            node_2="neurons",
            metadata={"chunk_id": "chunk_1", "confidence": 0.8},
        ),
        Triplet(
            node_1="amyloid",
            edge="forms",
            node_2="plaques",
            metadata={"chunk_id": "chunk_3"},
        ),
    ]


def test_knowledge_graph_creation():
    """Test creating a KnowledgeGraph instance."""
    kg = KnowledgeGraph(scientific_domain="test_domain")
    assert kg is not None
    assert kg.has_graph() is False
    assert len(kg.triplets) == 0
    assert hasattr(kg, "_backend")  # Backend should be available immediately


def test_knowledge_graph_from_triplets(sample_triplets_with_duplicates):
    """Test creating a KnowledgeGraph from a list of triplets."""
    kg = KnowledgeGraph.from_triplets(
        sample_triplets_with_duplicates, scientific_domain="test_domain"
    )

    # Basic graph structure checks
    assert kg is not None
    assert kg.has_graph() is True
    assert len(kg.triplets) == 4
    assert (
        len(kg.get_nodes()) == 6
    )  # microglia, activation, tau, neurons, amyloid, plaques
    assert len(kg.get_edges()) == 3  # undergoes, accumulates, forms

    # Check sources attribute is present on all nodes
    for node in kg.get_nodes():
        node_attrs = kg.get_node_attrs(node)
        assert "sources" in node_attrs
        assert isinstance(node_attrs["sources"], list)

    # Check sources attribute is present on all edges
    for source, target, data in kg.get_edges_data():
        assert "sources" in data
        assert isinstance(data["sources"], list)
        assert "relation" in data


def test_metadata_tracking(sample_triplets_with_duplicates):
    """Test that metadata from triplets is tracked in nodes and edges."""
    kg = KnowledgeGraph.from_triplets(
        sample_triplets_with_duplicates, scientific_domain="test_domain"
    )

    # Check that nodes have sources lists
    for node in kg.get_nodes():
        node_attrs = kg.get_node_attrs(node)
        sources = node_attrs["sources"]
        assert isinstance(sources, list)
        assert len(sources) > 0

    # Check specific node metadata
    microglia_attrs = kg.get_node_attrs("microglia")
    microglia_sources = microglia_attrs["sources"]
    assert len(microglia_sources) == 2  # Appears in 2 triplets

    # Check that all source metadata contains required fields
    for source in microglia_sources:
        assert "relation" in source
        assert "triplet_id" in source

    # Check edge metadata
    edges_data = kg.get_edge_attrs("microglia", "activation")
    for edge_data in edges_data:
        assert "sources" in edge_data
        assert len(edge_data["sources"]) == 2  # Two triplets for this edge

    # Check that edge sources contain metadata from both triplets
    source_chunk_ids = [
        source.get("chunk_id")
        for source in edge_data["sources"]
        if "chunk_id" in source
    ]
    assert "chunk_1" in source_chunk_ids
    assert "chunk_2" in source_chunk_ids

    # Check that metadata from both triplets is preserved
    has_confidence = False
    has_source = False
    for source in edge_data["sources"]:
        if "confidence" in source:
            has_confidence = True
        if "source" in source:
            has_source = True
    assert has_confidence
    assert has_source


def test_triplets_property():
    """Test that the triplets property correctly reconstructs triplets from the graph."""
    # Create triplets for testing
    original_triplets = [
        Triplet(
            node_1="protein_A",
            edge="binds_to",
            node_2="receptor_X",
            metadata={"source": "paper_1", "confidence": 0.9},
        ),
        Triplet(
            node_1="protein_B",
            edge="activates",
            node_2="enzyme_Y",
            metadata={"source": "paper_2", "confidence": 0.8},
        ),
    ]

    kg = KnowledgeGraph.from_triplets(
        original_triplets, scientific_domain="test_domain"
    )

    # Get triplets back from the graph
    reconstructed_triplets = kg.triplets

    # Check that we have the same number of triplets
    assert len(reconstructed_triplets) == len(original_triplets)

    # Check that triplets contain the correct data
    # Create dictionaries for easier comparison
    original_data = {}
    for t in original_triplets:
        key = (t.node_1, t.edge, t.node_2)
        original_data[key] = t.metadata

    reconstructed_data = {}
    for t in reconstructed_triplets:
        key = (t.node_1, t.edge, t.node_2)
        reconstructed_data[key] = t.metadata

    # Check keys match
    assert set(original_data.keys()) == set(reconstructed_data.keys())

    # Check metadata is preserved
    for key in original_data:
        for meta_key, meta_value in original_data[key].items():
            assert meta_key in reconstructed_data[key]
            assert reconstructed_data[key][meta_key] == meta_value


def test_save_and_load_file():
    """Test saving and loading a knowledge graph to/from a file."""
    # Create triplets for testing
    original_triplets = [
        Triplet(
            node_1="protein_A",
            edge="binds_to",
            node_2="receptor_X",
            metadata={"source": "paper_1", "confidence": 0.9},
        ),
        Triplet(
            node_1="protein_B",
            edge="activates",
            node_2="enzyme_Y",
            metadata={"source": "paper_2", "confidence": 0.8},
        ),
    ]

    # Create a knowledge graph
    kg = KnowledgeGraph.from_triplets(
        original_triplets, scientific_domain="test_domain"
    )

    # Add custom config
    kg.config = {"custom_setting": "test_value"}

    # Create a temporary file for testing
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as temp_file:
        temp_filename = temp_file.name

    try:
        # Save the graph to the temporary file
        kg.save_to_file(temp_filename)

        # Verify the file exists
        assert os.path.exists(temp_filename)

        # Verify file contents are valid JSON
        with open(temp_filename, "r", encoding="utf-8") as f:
            json_content = f.read()
            assert json_content  # File should not be empty
            import json

            data = json.loads(json_content)  # Should parse without errors
            assert "graph" in data
            assert "config" in data

        # Load the graph from the file
        loaded_kg = KnowledgeGraph.load_from_file(
            temp_filename, scientific_domain="test_domain"
        )

        # Check that the loaded graph has the same structure
        assert loaded_kg.has_graph()
        assert len(loaded_kg.get_nodes()) == len(kg.get_nodes())
        assert len(loaded_kg.get_edges()) == len(kg.get_edges())
        assert len(loaded_kg.triplets) == len(kg.triplets)

        # Check that configs are preserved
        assert loaded_kg.config == kg.config
        assert loaded_kg.config["custom_setting"] == "test_value"

        # Check that nodes match
        assert set(loaded_kg.get_nodes()) == set(kg.get_nodes())

        # Check that edges match
        loaded_edges = []
        for source, target, data in loaded_kg.get_edges_data():
            loaded_edges.append((source, target, data["relation"]))

        original_edges = []
        for source, target, data in kg.get_edges_data():
            original_edges.append((source, target, data["relation"]))

        assert set(loaded_edges) == set(original_edges)

    finally:
        # Clean up the temporary file
        if os.path.exists(temp_filename):
            os.remove(temp_filename)


def test_load_from_nonexistent_file():
    """Test loading from a non-existent file raises the correct exception."""
    # Choose a file path that definitely doesn't exist
    nonexistent_file = "/nonexistent/path/to/graph.json"

    # Check that attempting to load it raises a FileNotFoundError
    with pytest.raises(FileNotFoundError):
        KnowledgeGraph.load_from_file(nonexistent_file, scientific_domain="test_domain")


def test_save_empty_graph():
    """Test that saving an empty graph raises the correct exception."""
    # Create an empty knowledge graph
    kg = KnowledgeGraph(scientific_domain="test_domain")

    # Create a temporary file for testing
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as temp_file:
        temp_filename = temp_file.name

    try:
        # Check that attempting to save it raises a ValueError
        with pytest.raises(ValueError, match="Cannot save an empty knowledge graph"):
            kg.save_to_file(temp_filename)
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_filename):
            os.unlink(temp_filename)


def test_random_walk():
    """Test the random walk functionality in KnowledgeGraph."""
    # Create a simple graph for testing random walks
    triplets = [
        Triplet(node_1="A", edge="connects_to", node_2="B"),
        Triplet(node_1="B", edge="connects_to", node_2="C"),
        Triplet(node_1="C", edge="connects_to", node_2="D"),
        Triplet(node_1="A", edge="connects_to", node_2="E"),
        Triplet(node_1="E", edge="connects_to", node_2="F"),
    ]
    kg = KnowledgeGraph.from_triplets(triplets, scientific_domain="test_domain")

    # Test basic random walk functionality
    walk = kg.random_walk("A", max_steps=3)
    assert walk[0] == "A"  # Walk should start at the specified node
    assert len(walk) <= 4  # Walk should have at most 4 nodes (start + up to 3 steps)

    # Test random walk with max_steps=0 (should just return start node)
    walk = kg.random_walk("B", max_steps=0)
    assert walk == ["B"]

    # Test that error is raised for non-existent node
    with pytest.raises(
        ValueError, match="Start node 'non_existent' not found in the graph"
    ):
        kg.random_walk("non_existent")

    # Run multiple walks and verify deterministic behavior
    # Set a seed for reproducibility
    import random

    random.seed(42)

    walk1 = kg.random_walk("A", max_steps=10)

    # Reset the seed for identical result
    random.seed(42)
    walk2 = kg.random_walk("A", max_steps=10)

    # Walks should be identical with the same seed
    assert walk1 == walk2
