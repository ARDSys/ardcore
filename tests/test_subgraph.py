import os
import random
import tempfile
from unittest.mock import MagicMock, patch

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pytest
from langchain_core.prompts import PromptTemplate

from ardcore.data.triplets import Triplet
from ardcore.knowledge_graph import KnowledgeGraph
from ardcore.subgraph.subgraph import Subgraph
from ardcore.subgraph.subgraph_generator import (
    SingleNodeSubgraphGenerator,
    SubgraphGenerator,
)
from ardcore.subgraph.subgraph_generator.embedding import EmbeddingPathGenerator
from ardcore.subgraph.subgraph_generator.llm_walk import LLMWalkGenerator
from ardcore.subgraph.subgraph_generator.random_walk import (
    RandomWalkGenerator,
)
from ardcore.subgraph.subgraph_generator.randomized_embedding import (
    RandomizedEmbeddingPathGenerator,
)
from ardcore.subgraph.subgraph_generator.shortest_path import ShortestPathGenerator


def test_subgraph_creation(sample_knowledge_graph):
    """Test creating a Subgraph from a KnowledgeGraph."""
    # Create a subgraph from A to D
    subgraph = Subgraph.from_two_nodes(
        sample_knowledge_graph, "A", "D", ShortestPathGenerator()
    )

    # Check that the subgraph was created successfully
    assert subgraph is not None
    assert subgraph.has_graph()

    # Check start and end nodes
    assert subgraph.start_node == "A"
    assert subgraph.end_node == "D"

    # Check that the path exists
    assert len(subgraph.path_nodes) > 0
    assert subgraph.path_nodes[0] == "A"
    assert subgraph.path_nodes[-1] == "D"


def test_include_neighbors(sample_knowledge_graph):
    """Test including neighbors with different probabilities and max_nodes limits."""
    # Create a subgraph with no neighbors (probability = 0.0)
    subgraph_no_neighbors = Subgraph.from_two_nodes(
        sample_knowledge_graph,
        "A",
        "D",
        ShortestPathGenerator(),
        neighbor_probability=0.0,
    )

    # Should only include nodes on the path
    assert set(subgraph_no_neighbors.get_nodes()) == set(
        subgraph_no_neighbors.path_nodes
    )

    # Create a subgraph with all neighbors (probability = 1.0)
    subgraph_all_neighbors = Subgraph.from_two_nodes(
        sample_knowledge_graph,
        "A",
        "D",
        ShortestPathGenerator(),
        neighbor_probability=1.0,
    )

    # Should include more nodes than just the path
    assert len(subgraph_all_neighbors.get_nodes()) >= len(
        subgraph_all_neighbors.path_nodes
    )

    # Create a subgraph with max_nodes limit
    path_length = len(
        Subgraph.from_two_nodes(
            sample_knowledge_graph, "A", "D", ShortestPathGenerator()
        ).path_nodes
    )
    subgraph_limited = Subgraph.from_two_nodes(
        sample_knowledge_graph,
        "A",
        "D",
        ShortestPathGenerator(),
        neighbor_probability=1.0,
        max_nodes=path_length + 1,
    )

    # Should have at most path_length + 1 nodes
    assert len(subgraph_limited.get_nodes()) <= path_length + 1

    # All path nodes should be included
    for node in subgraph_limited.path_nodes:
        assert node in subgraph_limited.get_nodes()


def test_neighbor_probability_validation(sample_knowledge_graph):
    """Test validation of neighbor_probability parameter."""
    # Test with invalid probability values
    with pytest.raises(ValueError):
        Subgraph.from_two_nodes(
            sample_knowledge_graph,
            "A",
            "D",
            ShortestPathGenerator(),
            neighbor_probability=-0.1,
        )

    with pytest.raises(ValueError):
        Subgraph.from_two_nodes(
            sample_knowledge_graph,
            "A",
            "D",
            ShortestPathGenerator(),
            neighbor_probability=1.1,
        )

    # Test with valid probability values
    Subgraph.from_two_nodes(
        sample_knowledge_graph,
        "A",
        "D",
        ShortestPathGenerator(),
        neighbor_probability=0.0,
    )
    Subgraph.from_two_nodes(
        sample_knowledge_graph,
        "A",
        "D",
        ShortestPathGenerator(),
        neighbor_probability=0.5,
    )
    Subgraph.from_two_nodes(
        sample_knowledge_graph,
        "A",
        "D",
        ShortestPathGenerator(),
        neighbor_probability=1.0,
    )


def test_path_edges(sample_knowledge_graph):
    """Test getting the edges on the path."""
    # Create a subgraph from A to D
    subgraph = Subgraph.from_two_nodes(
        sample_knowledge_graph, "A", "D", ShortestPathGenerator()
    )

    # Get the path edges
    path_edges = subgraph.get_path_edges()

    # Check that the path edges are correct
    assert len(path_edges) == 2
    assert path_edges[0][0] == "A"  # source
    assert path_edges[0][2] == "E"  # target
    assert path_edges[1][0] == "E"  # source
    assert path_edges[1][2] == "D"  # target


def test_invalid_nodes(sample_knowledge_graph):
    """Test that creating a subgraph with invalid nodes raises an error."""
    # Test with invalid start node
    with pytest.raises(ValueError, match="Start node 'Z' not found in the graph"):
        Subgraph.from_two_nodes(
            sample_knowledge_graph, "Z", "D", ShortestPathGenerator()
        )

    # Test with invalid end node
    with pytest.raises(ValueError, match="End node 'Z' not found in the graph"):
        Subgraph.from_two_nodes(
            sample_knowledge_graph, "A", "Z", ShortestPathGenerator()
        )


def test_no_path(sample_knowledge_graph):
    """Test that creating a subgraph with no path raises an error."""
    # Add an isolated node to the graph
    isolated_triplet = Triplet(node_1="X", edge="isolated", node_2="Y")
    kg = KnowledgeGraph.from_triplets(
        [isolated_triplet], scientific_domain="test_domain"
    )

    # Add all triplets from the sample graph
    for triplet in sample_knowledge_graph.triplets:
        kg.add_triplets([triplet])

    # Test with nodes that have no path between them
    with pytest.raises(nx.NetworkXNoPath):
        Subgraph.from_two_nodes(kg, "A", "X", ShortestPathGenerator())


def test_string_representation(sample_knowledge_graph):
    """Test the string representation of a subgraph."""
    # Create a subgraph from A to D
    subgraph = Subgraph.from_two_nodes(
        sample_knowledge_graph, "A", "D", ShortestPathGenerator()
    )

    # Get the string representation
    string_repr = str(subgraph)

    # Check that it contains the path
    assert "Path: A -> E -> D" in string_repr


def test_visualization(sample_knowledge_graph):
    """Test the visualization of a subgraph."""
    # Create a subgraph from A to D
    subgraph = Subgraph.from_two_nodes(
        sample_knowledge_graph, "A", "D", ShortestPathGenerator()
    )

    # Create a temporary file for saving the visualization
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
        temp_filename = temp_file.name

    try:
        # Test visualization with default parameters
        fig, ax = subgraph.visualize()
        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)
        plt.close(fig)

        # Test visualization with custom parameters
        fig, ax = subgraph.visualize(
            figsize=(8, 6),
            node_size=500,
            font_size=8,
            edge_label_font_size=6,
            title="Test Visualization",
            layout="circular",
            save_path=temp_filename,
        )
        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)
        plt.close(fig)

        # Check that the file was created
        assert os.path.exists(temp_filename)
        assert os.path.getsize(temp_filename) > 0

    finally:
        # Clean up the temporary file
        if os.path.exists(temp_filename):
            os.unlink(temp_filename)
        plt.close("all")


def test_to_cypher_string(sample_knowledge_graph):
    """Test the Cypher string representation of a subgraph."""
    # Create a subgraph from A to D
    subgraph = Subgraph.from_two_nodes(
        sample_knowledge_graph, "A", "D", ShortestPathGenerator()
    )

    # Get the Cypher string
    cypher_str = subgraph.to_cypher_string()

    # Check that it contains the path edges
    assert "(A)-[:relates_to]->(E)" in cypher_str
    assert "(E)-[:influences]->(D)" in cypher_str

    # Check formatting
    assert "," in cypher_str  # Patterns should be comma-separated
    assert "\n" in cypher_str  # Should have line breaks for readability


def test_get_random_node(sample_knowledge_graph):
    """Test getting a random node from the graph."""
    # Get a random node multiple times to ensure randomness
    nodes = set()
    for _ in range(50):  # Try enough times to likely get different nodes
        node = sample_knowledge_graph.get_random_node()
        assert node is not None  # Should not return None for non-empty graph
        assert node in sample_knowledge_graph.get_nodes()  # Should be a valid node
        nodes.add(node)

    # Should have gotten at least a few different nodes
    assert len(nodes) > 1, "Random selection should return different nodes"

    # Test with empty graph
    empty_kg = KnowledgeGraph(scientific_domain="test_domain")
    assert empty_kg.get_random_node() is None


def test_serialization(sample_knowledge_graph):
    """Test saving and loading a subgraph to/from a file."""
    # Create a subgraph
    subgraph = Subgraph.from_two_nodes(
        sample_knowledge_graph, "A", "D", ShortestPathGenerator()
    )

    # Add context to the subgraph
    mock_context = "This is a mock analysis of the subgraph."
    subgraph._context = mock_context

    # Create a temporary file for testing
    with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as temp_file:
        temp_filename = temp_file.name

    try:
        # Save the subgraph to the file
        subgraph.save_to_file(temp_filename)

        # Load the subgraph from the file
        loaded_subgraph = Subgraph.load_from_file(
            temp_filename, scientific_domain="test_domain"
        )

        # Check that the loaded subgraph has the same attributes
        assert loaded_subgraph.start_node == subgraph.start_node
        assert loaded_subgraph.end_node == subgraph.end_node
        assert loaded_subgraph.path_nodes == subgraph.path_nodes
        assert loaded_subgraph.context == mock_context

        # Check that the graph structure is preserved
        assert set(loaded_subgraph.get_nodes()) == set(subgraph.get_nodes())
        assert len(loaded_subgraph.get_path_edges()) == len(subgraph.get_path_edges())

        # Check that the original graph is represented as a placeholder
        assert loaded_subgraph.original_graph is not None
        assert isinstance(loaded_subgraph.original_graph, KnowledgeGraph)

        # The original graph should be a copy of the subgraph itself
        assert len(loaded_subgraph.original_graph.get_nodes()) == len(
            subgraph.get_nodes()
        )

    finally:
        # Clean up the temporary file
        if os.path.exists(temp_filename):
            os.remove(temp_filename)


def test_to_json(sample_knowledge_graph):
    """Test the to_json method of the Subgraph class."""
    # Create a subgraph
    subgraph = Subgraph.from_two_nodes(
        sample_knowledge_graph, "A", "D", ShortestPathGenerator()
    )

    # Add context and scores to the subgraph
    mock_context = "This is a mock analysis of the subgraph."
    subgraph._context = mock_context
    subgraph._path_score = 4
    subgraph._path_score_justification = "The path has a logical connection."

    # Get the JSON representation
    json_data = subgraph.to_json()

    # Check that the JSON data has the expected structure and content
    assert isinstance(json_data, dict)

    # Check basic metadata
    assert json_data["start_node"] == "A"
    assert json_data["end_node"] == "D"
    assert json_data["path_nodes"] == subgraph.path_nodes
    assert json_data["context"] == mock_context
    assert json_data["path_score"] == 4
    assert json_data["path_score_justification"] == "The path has a logical connection."

    # Check graph stats
    assert "graph_stats" in json_data
    assert json_data["graph_stats"]["node_count"] == len(subgraph.get_nodes())
    assert json_data["graph_stats"]["edge_count"] == len(list(subgraph.get_edges()))
    assert json_data["graph_stats"]["path_length"] == len(subgraph.path_nodes)

    # Check path edges
    assert "path_edges" in json_data
    path_edges = json_data["path_edges"]
    assert len(path_edges) == len(subgraph.path_nodes) - 1

    # Check that each path edge has the expected structure
    for edge in path_edges:
        assert "source" in edge
        assert "target" in edge
        assert "relation" in edge

    # Check original graph metadata
    assert "original_graph_metadata" in json_data
    assert json_data["original_graph_metadata"]["node_count"] == len(
        subgraph.original_graph.get_nodes()
    )


def test_serialization_with_nonexistent_file():
    """Test loading a subgraph from a nonexistent file."""
    with pytest.raises(FileNotFoundError):
        Subgraph.load_from_file(
            "nonexistent_file.json", scientific_domain="test_domain"
        )


def test_contextualize(sample_knowledge_graph):
    """Test the contextualize method with a mock LLM."""
    # Create a mock LLM
    mock_llm = MagicMock()
    mock_llm.return_value = MagicMock(
        content="This is a mock analysis of the subgraph."
    )

    # Create a subgraph
    subgraph = Subgraph.from_two_nodes(
        sample_knowledge_graph, "A", "D", ShortestPathGenerator()
    )

    # Check that context is initially None
    assert subgraph.context is None

    # Test with default parameters
    analysis = subgraph.contextualize(mock_llm)
    assert analysis == "This is a mock analysis of the subgraph."

    # Check that context was set
    assert subgraph.context == "This is a mock analysis of the subgraph."

    # Verify the LLM was called with the correct prompt
    mock_llm.assert_called_once()
    call_args = mock_llm.call_args[0][0]
    assert "A" in call_args
    assert "D" in call_args
    assert "(A)-[:relates_to]->(E)" in call_args

    # Reset mock
    mock_llm.reset_mock()

    # Test with custom prompt
    custom_prompt = PromptTemplate.from_template(
        "Custom prompt: {graph_str} from {start_node} to {end_node}"
    )
    analysis = subgraph.contextualize(mock_llm, prompt=custom_prompt)

    # Verify custom prompt was used and context was updated
    mock_llm.assert_called_once()
    call_args = mock_llm.call_args[0][0]
    assert "Custom prompt:" in call_args
    assert subgraph.context == "This is a mock analysis of the subgraph."


def test_subgraph_constructor(sample_knowledge_graph):
    """Test the Subgraph constructor directly."""
    # Use a known path for testing
    path_nodes = ["A", "B", "C", "D"]

    # Test constructor with valid parameters
    subgraph = Subgraph(
        original_graph=sample_knowledge_graph,
        start_node="A",
        end_node="D",
        path_nodes=path_nodes,
    )

    # Verify basic properties
    assert subgraph.start_node == "A"
    assert subgraph.end_node == "D"
    assert subgraph.path_nodes == path_nodes
    assert subgraph.original_graph == sample_knowledge_graph
    assert subgraph.subgraph_id is not None  # Check that an ID was generated

    # Test constructor with invalid start node
    with pytest.raises(ValueError, match="Start node 'Z' not found in the graph"):
        Subgraph(
            original_graph=sample_knowledge_graph,
            start_node="Z",
            end_node="D",
            path_nodes=["Z", "A", "D"],
        )

    # Test constructor with invalid end node
    with pytest.raises(ValueError, match="End node 'Z' not found in the graph"):
        Subgraph(
            original_graph=sample_knowledge_graph,
            start_node="A",
            end_node="Z",
            path_nodes=["A", "B", "Z"],
        )

    # Test constructor with invalid path nodes
    with pytest.raises(ValueError, match="Path nodes"):
        Subgraph(
            original_graph=sample_knowledge_graph,
            start_node="A",
            end_node="D",
            path_nodes=["A", "Z", "D"],  # Z does not exist in the graph
        )


class TestSubgraphGenerator:
    """Tests for the SubgraphGenerator base class and implementations."""

    def test_validate_nodes(self, sample_knowledge_graph):
        """Test the validate_nodes method in the base class."""
        generator = ShortestPathGenerator()

        # Valid nodes should not raise an exception
        generator.validate_nodes(sample_knowledge_graph, "A", "D")

        # Invalid start node should raise ValueError
        with pytest.raises(ValueError, match="Start node 'X' not found in the graph"):
            generator.validate_nodes(sample_knowledge_graph, "X", "D")

        # Invalid end node should raise ValueError
        with pytest.raises(ValueError, match="End node 'Z' not found in the graph"):
            generator.validate_nodes(sample_knowledge_graph, "A", "Z")

    def test_shortest_path_generator(self, sample_knowledge_graph):
        """Test the ShortestPathGenerator implementation."""
        generator = ShortestPathGenerator()

        # Generate path from A to D
        path_nodes = generator.generate_path_nodes(sample_knowledge_graph, "A", "D")

        # Check the path
        assert path_nodes is not None
        assert len(path_nodes) > 0
        assert path_nodes[0] == "A"
        assert path_nodes[-1] == "D"

        # Path should be the shortest (A -> E -> D)
        assert len(path_nodes) == 3

        # Check that path exists in the graph
        for i in range(len(path_nodes) - 1):
            source = path_nodes[i]
            target = path_nodes[i + 1]
            # Check if there's an edge in either direction
            assert sample_knowledge_graph.graph.has_edge(
                source,
                target,
                scientific_domain=sample_knowledge_graph.scientific_domain,
            ) or sample_knowledge_graph.graph.has_edge(
                target,
                source,
                scientific_domain=sample_knowledge_graph.scientific_domain,
            )

        # Test with invalid end node - should raise ValueError
        with pytest.raises(ValueError, match="End node 'Z' not found in the graph"):
            generator.generate_path_nodes(sample_knowledge_graph, "A", "Z")

        # Test with valid nodes but no path - mock the validation to pass but force NetworkXNoPath
        with patch.object(generator, "validate_nodes"):  # Skip validation
            # This should now raise NetworkXNoPath since we patched the validation
            with pytest.raises(nx.NetworkXNoPath):
                # Create a small disconnected graph where no path exists
                disconnected_triplets = [
                    Triplet(
                        node_1="P", edge="connects_to", node_2="Q"
                    ),  # Disconnected component 1
                    Triplet(
                        node_1="R", edge="connects_to", node_2="S"
                    ),  # Disconnected component 2
                ]
                disconnected_graph = KnowledgeGraph.from_triplets(
                    disconnected_triplets, scientific_domain="test_domain"
                )
                generator.generate_path_nodes(
                    disconnected_graph, "P", "S"
                )  # No path exists


def test_from_two_nodes_method(sample_knowledge_graph):
    """Test the from_two_nodes method."""
    # Create a ShortestPathGenerator
    generator = ShortestPathGenerator()

    # Use from_two_nodes with the generator
    subgraph = Subgraph.from_two_nodes(
        sample_knowledge_graph, "A", "D", generator, neighbor_probability=0.5
    )

    # Check that the subgraph was created successfully
    assert subgraph is not None
    assert subgraph.has_graph()

    # Check start and end nodes
    assert subgraph.start_node == "A"
    assert subgraph.end_node == "D"

    # Check that the path exists
    assert len(subgraph.path_nodes) > 0
    assert subgraph.path_nodes[0] == "A"
    assert subgraph.path_nodes[-1] == "D"

    # Check that the original graph is stored
    assert subgraph.original_graph is sample_knowledge_graph


def test_custom_subgraph_generator(sample_knowledge_graph):
    """Test creating a custom SubgraphGenerator implementation."""

    # Define a custom SubgraphGenerator
    class CustomPathGenerator(SubgraphGenerator):
        """A custom generator that always returns the same hardcoded path."""

        def generate_path_nodes(
            self, knowledge_graph: KnowledgeGraph, start_node: str, end_node: str
        ) -> list:
            """Generate a hardcoded path for testing."""
            # Validate nodes first (inherited from base class)
            self.validate_nodes(knowledge_graph, start_node, end_node)

            # For this test, always return path through node E
            if start_node == "A" and end_node == "D":
                return ["A", "E", "D"]

            # Fallback to a simple direct path if valid
            if knowledge_graph.graph.has_edge(start_node, end_node):
                return [start_node, end_node]

            # No path found
            raise nx.NetworkXNoPath(
                f"No path exists between '{start_node}' and '{end_node}'"
            )

    # Create the custom generator
    custom_generator = CustomPathGenerator()

    # Use from_two_nodes with the custom generator
    subgraph = Subgraph.from_two_nodes(
        sample_knowledge_graph, "A", "D", custom_generator
    )

    # Check that the custom path was used
    assert subgraph.path_nodes == ["A", "E", "D"]

    # Check that the subgraph contains the expected nodes
    assert subgraph.has_node("A")
    assert subgraph.has_node("E")
    assert subgraph.has_node("D")


def test_single_node_subgraph_generator_interface(sample_knowledge_graph):
    """Test the SingleNodeSubgraphGenerator interface."""

    # Define a test implementation of SingleNodeSubgraphGenerator
    class TestSingleNodeGenerator(SingleNodeSubgraphGenerator):
        """A test implementation that returns a fixed path."""

        def generate_path_nodes(
            self, knowledge_graph: KnowledgeGraph, start_node: str
        ) -> list:
            """Generate a hardcoded path for testing."""
            # Validate the node first
            self.validate_node(knowledge_graph, start_node)

            # For testing, return a simple path
            if start_node == "A":
                return ["A", "B", "C"]
            elif start_node == "B":
                return ["B", "C"]
            else:
                return [start_node]

    # Create the test generator
    test_generator = TestSingleNodeGenerator()

    # Test validate_node method
    test_generator.validate_node(sample_knowledge_graph, "A")  # Should not raise error

    with pytest.raises(ValueError, match="Start node 'Z' not found in the graph"):
        test_generator.validate_node(sample_knowledge_graph, "Z")

    # Test generate_path_nodes method
    path = test_generator.generate_path_nodes(sample_knowledge_graph, "A")
    assert path == ["A", "B", "C"]

    path = test_generator.generate_path_nodes(sample_knowledge_graph, "B")
    assert path == ["B", "C"]

    path = test_generator.generate_path_nodes(sample_knowledge_graph, "E")
    assert path == ["E"]


def test_from_one_node_method(sample_knowledge_graph):
    """Test the from_one_node method of the Subgraph class."""

    # Define a custom SingleNodeSubgraphGenerator for testing
    class FixedPathGenerator(SingleNodeSubgraphGenerator):
        """Returns a fixed path starting from the given node."""

        def generate_path_nodes(
            self, knowledge_graph: KnowledgeGraph, start_node: str
        ) -> list:
            """Generate a fixed path based on the start node."""
            self.validate_node(knowledge_graph, start_node)

            if start_node == "A":
                return ["A", "B", "C", "D"]
            elif start_node == "B":
                return ["B", "C"]
            else:
                return [start_node]

    # Create a subgraph using from_one_node with our fixed path generator
    subgraph = Subgraph.from_one_node(
        sample_knowledge_graph,
        "A",
        FixedPathGenerator(),
        neighbor_probability=0.5,
        max_nodes=10,
    )

    # Verify the subgraph properties
    assert subgraph.start_node == "A"
    assert subgraph.end_node == "D"  # Last node in our fixed path
    assert subgraph.path_nodes == ["A", "B", "C", "D"]

    # Test without neighbors
    subgraph_no_neighbors = Subgraph.from_one_node(
        sample_knowledge_graph, "A", FixedPathGenerator(), neighbor_probability=0.0
    )

    # Should only contain path nodes
    assert set(subgraph_no_neighbors.get_nodes()) == set(
        subgraph_no_neighbors.path_nodes
    )

    # Test with max_nodes limit
    subgraph_limited = Subgraph.from_one_node(
        sample_knowledge_graph,
        "A",
        FixedPathGenerator(),
        neighbor_probability=1.0,
        max_nodes=4,
    )

    # Should have at most 3 nodes, and they should be path nodes
    assert len(subgraph_limited.get_nodes()) == 4
    for node in subgraph_limited.get_nodes():
        assert node in ["A", "B", "C", "D"]

    # Test with empty path
    class EmptyPathGenerator(SingleNodeSubgraphGenerator):
        def generate_path_nodes(
            self, knowledge_graph: KnowledgeGraph, start_node: str
        ) -> list:
            self.validate_node(knowledge_graph, start_node)
            return []

    with pytest.raises(ValueError, match="Empty path generated from node 'A'"):
        Subgraph.from_one_node(sample_knowledge_graph, "A", EmptyPathGenerator())

    # Test with invalid start node
    with pytest.raises(ValueError, match="Start node 'Z' not found in the graph"):
        Subgraph.from_one_node(sample_knowledge_graph, "Z", FixedPathGenerator())


def test_random_walk_generator(sample_knowledge_graph):
    """Test the RandomWalkGenerator implementation."""
    # Create a RandomWalkGenerator with a fixed seed for reproducibility
    generator = RandomWalkGenerator(max_steps=10, seed=42)

    # Try to generate a path from A to D
    try:
        path_nodes = generator.generate_path_nodes(sample_knowledge_graph, "A", "D")

        # Check that the path begins with A
        assert path_nodes[0] == "A"

        # Check that the path either contains D or ended with a NetworkXNoPath exception
        if "D" in path_nodes:
            assert path_nodes.index("D") == len(path_nodes) - 1

        # Check that all nodes in the path are connected in the graph
        for i in range(len(path_nodes) - 1):
            source = path_nodes[i]
            target = path_nodes[i + 1]
            assert sample_knowledge_graph.graph.has_edge(
                source,
                target,
                scientific_domain=sample_knowledge_graph.scientific_domain,
            ) or sample_knowledge_graph.graph.has_edge(
                target,
                source,
                scientific_domain=sample_knowledge_graph.scientific_domain,
            )

    except nx.NetworkXNoPath:
        # It's possible that the random walk didn't find a path, which is acceptable
        pass

    # Test with invalid node - should raise ValueError
    with pytest.raises(
        ValueError, match="End node 'NonExistentNode' not found in the graph"
    ):
        generator.generate_path_nodes(sample_knowledge_graph, "A", "NonExistentNode")

    # Test with a seed that ensures a path is found (assuming there is a reliable path)
    # Create a more connected graph to make sure a path exists
    connected_triplets = [
        # Create a fully connected small graph
        Triplet(node_1="X", edge="connects_to", node_2="Y"),
        Triplet(node_1="Y", edge="connects_to", node_2="Z"),
        Triplet(node_1="Z", edge="connects_to", node_2="X"),
    ]
    connected_graph = KnowledgeGraph.from_triplets(
        connected_triplets, scientific_domain="test_domain"
    )

    # With this fully connected graph, we should find a path
    deterministic_generator = RandomWalkGenerator(max_steps=5, seed=42)
    path = deterministic_generator.generate_path_nodes(connected_graph, "X", "Z")

    # Check the path
    assert path[0] == "X"
    assert "Z" in path

    # Test using the generator with from_two_nodes
    try:
        subgraph = Subgraph.from_two_nodes(
            sample_knowledge_graph,
            "A",
            "D",
            RandomWalkGenerator(
                max_steps=15, seed=42
            ),  # Use more steps to increase chance of finding a path
        )

        # Check the subgraph properties
        assert subgraph.start_node == "A"
        assert subgraph.end_node == "D"
        assert len(subgraph.path_nodes) > 0
    except nx.NetworkXNoPath:
        # It's possible that the random walk didn't find a path, which is acceptable
        pass


def test_embedding_path_generator():
    """Test the EmbeddingPathGenerator implementation."""
    # Create a simple graph
    triplets = [
        Triplet(node_1="A", edge="connects_to", node_2="B"),
        Triplet(node_1="B", edge="leads_to", node_2="C"),
        Triplet(node_1="C", edge="results_in", node_2="D"),
        Triplet(node_1="A", edge="relates_to", node_2="E"),
        Triplet(node_1="E", edge="influences", node_2="D"),
    ]
    knowledge_graph = KnowledgeGraph.from_triplets(
        triplets, scientific_domain="test_domain"
    )

    # Create mock embeddings for the nodes
    # In a real scenario, these would be generated by an embedding model
    node_embeddings = {
        "A": np.array([0.1, 0.2, 0.3]),
        "B": np.array([0.2, 0.3, 0.4]),
        "C": np.array([0.3, 0.4, 0.5]),
        "D": np.array([0.4, 0.5, 0.6]),
        "E": np.array([0.5, 0.1, 0.2]),
    }

    # Create an EmbeddingPathGenerator with a fixed seed for reproducibility
    generator = EmbeddingPathGenerator(
        node_embeddings=node_embeddings, top_k=2, seed=42
    )

    # Generate a path from A to D
    path_nodes = generator.generate_path_nodes(knowledge_graph, "A", "D")

    # Check that the path begins with A and ends with D
    assert path_nodes[0] == "A"
    assert path_nodes[-1] == "D"

    # Check that all nodes in the path are connected in the graph
    for i in range(len(path_nodes) - 1):
        source = path_nodes[i]
        target = path_nodes[i + 1]
        assert knowledge_graph.graph.has_edge(
            source, target, scientific_domain=knowledge_graph.scientific_domain
        ) or knowledge_graph.graph.has_edge(
            target, source, scientific_domain=knowledge_graph.scientific_domain
        )

    # Test with invalid node - should raise ValueError
    with pytest.raises(ValueError, match="End node 'Z' not found in the graph"):
        generator.generate_path_nodes(knowledge_graph, "A", "Z")

    # Test using the generator with from_two_nodes
    subgraph = Subgraph.from_two_nodes(knowledge_graph, "A", "D", generator)

    # Check the subgraph properties
    assert subgraph.start_node == "A"
    assert subgraph.end_node == "D"
    assert len(subgraph.path_nodes) > 0
    assert subgraph.path_nodes[0] == "A"
    assert subgraph.path_nodes[-1] == "D"


def test_randomized_embedding_path_generator():
    """Test the RandomizedEmbeddingPathGenerator implementation."""
    # Create a simple graph
    triplets = [
        Triplet(node_1="A", edge="connects_to", node_2="B"),
        Triplet(node_1="B", edge="leads_to", node_2="C"),
        Triplet(node_1="C", edge="results_in", node_2="D"),
        Triplet(node_1="A", edge="relates_to", node_2="E"),
        Triplet(node_1="E", edge="influences", node_2="D"),
        # Add more connections for waypoints
        Triplet(node_1="B", edge="interacts_with", node_2="F"),
        Triplet(node_1="F", edge="affects", node_2="G"),
        Triplet(node_1="G", edge="connects_to", node_2="D"),
        Triplet(node_1="E", edge="relates_to", node_2="H"),
        Triplet(node_1="H", edge="leads_to", node_2="D"),
    ]
    knowledge_graph = KnowledgeGraph.from_triplets(
        triplets, scientific_domain="test_domain"
    )

    # Create mock embeddings for the nodes
    node_embeddings = {
        "A": np.array([0.1, 0.2, 0.3]),
        "B": np.array([0.2, 0.3, 0.4]),
        "C": np.array([0.3, 0.4, 0.5]),
        "D": np.array([0.4, 0.5, 0.6]),
        "E": np.array([0.5, 0.1, 0.2]),
        "F": np.array([0.6, 0.2, 0.3]),
        "G": np.array([0.7, 0.3, 0.4]),
        "H": np.array([0.8, 0.4, 0.5]),
    }

    # Test with no randomness (should behave like regular embedding path)
    generator_no_random = RandomizedEmbeddingPathGenerator(
        node_embeddings=node_embeddings,
        top_k=2,
        randomness_factor=0.0,
        num_random_waypoints=0,
        seed=42,
    )

    path_no_random = generator_no_random.generate_path_nodes(knowledge_graph, "A", "D")

    # Check that the path begins with A and ends with D
    assert path_no_random[0] == "A"
    assert path_no_random[-1] == "D"

    # Test with randomness but no waypoints
    generator_random = RandomizedEmbeddingPathGenerator(
        node_embeddings=node_embeddings,
        top_k=2,
        randomness_factor=0.5,
        num_random_waypoints=0,
        seed=42,
    )

    path_random = generator_random.generate_path_nodes(knowledge_graph, "A", "D")

    # Check that the path begins with A and ends with D
    assert path_random[0] == "A"
    assert path_random[-1] == "D"

    # Test with randomness and waypoints
    generator_with_waypoints = RandomizedEmbeddingPathGenerator(
        node_embeddings=node_embeddings,
        top_k=2,
        randomness_factor=0.5,
        num_random_waypoints=2,
        seed=42,
    )

    path_with_waypoints = generator_with_waypoints.generate_path_nodes(
        knowledge_graph, "A", "D"
    )

    # Check that the path begins with A and ends with D
    assert path_with_waypoints[0] == "A"
    assert path_with_waypoints[-1] == "D"

    # Check that all nodes in the path are connected in the graph
    for i in range(len(path_with_waypoints) - 1):
        source = path_with_waypoints[i]
        target = path_with_waypoints[i + 1]
        assert knowledge_graph.graph.has_edge(
            source, target, scientific_domain=knowledge_graph.scientific_domain
        ) or knowledge_graph.graph.has_edge(
            target, source, scientific_domain=knowledge_graph.scientific_domain
        )

    # Test with invalid node - should raise ValueError
    with pytest.raises(ValueError, match="End node 'Z' not found in the graph"):
        generator_with_waypoints.generate_path_nodes(knowledge_graph, "A", "Z")

    # Test using the generator with from_two_nodes
    subgraph = Subgraph.from_two_nodes(
        knowledge_graph, "A", "D", generator_with_waypoints
    )

    # Check the subgraph properties
    assert subgraph.start_node == "A"
    assert subgraph.end_node == "D"
    assert len(subgraph.path_nodes) > 0
    assert subgraph.path_nodes[0] == "A"
    assert subgraph.path_nodes[-1] == "D"


def test_llm_walk_generator(sample_knowledge_graph):
    """Test the LLMWalkGenerator implementation."""
    # Create a mock LLM response that always selects the first available neighbor
    mock_response = MagicMock()
    mock_response.content = "NEXT_NODE: B"  # This will select node B first

    mock_llm = MagicMock()
    mock_llm.return_value = mock_response

    # Create the generator
    generator = LLMWalkGenerator(max_steps=3, llm=mock_llm)

    # Test generate_path_nodes with our mock LLM
    path = generator.generate_path_nodes(sample_knowledge_graph, "A")

    # Verify basic path properties
    assert path[0] == "A"  # Should start with start node
    assert len(path) <= 4  # Start node + max 3 steps

    # Verify the LLM was called
    assert mock_llm.called
    # Extract arguments from all calls to the LLM
    call_args_list = [call[0][0] for call in mock_llm.call_args_list]

    # The first call should be for node A
    first_call_args = call_args_list[0]
    assert "Current node: A" in first_call_args
    assert "Current path:" in first_call_args
    assert "Available neighbors:" in first_call_args

    # Test different response handling
    # Update the mock to return a different node
    mock_response.content = "NEXT_NODE: E"

    # Reset the mock to track new calls
    mock_llm.reset_mock()

    # Generate another path
    path = generator.generate_path_nodes(sample_knowledge_graph, "A")

    # Verify the LLM was called
    assert mock_llm.called

    # Test creating a subgraph using the generator
    # Create a subgraph using from_one_node with the LLM generator
    subgraph = Subgraph.from_one_node(
        sample_knowledge_graph,
        "A",
        LLMWalkGenerator(max_steps=3, llm=mock_llm),
        neighbor_probability=0.5,
    )

    # Verify the subgraph properties
    assert subgraph.start_node == "A"
    assert subgraph.path_nodes[0] == "A"

    # Test with invalid start node
    with pytest.raises(ValueError, match="Start node 'Z' not found in the graph"):
        generator.generate_path_nodes(sample_knowledge_graph, "Z")

    # Test with node that has no neighbors
    isolated_graph = nx.DiGraph()
    isolated_graph.add_node("isolated")
    # Create a triplet for the isolated node
    isolated_triplet = Triplet(node_1="isolated", edge="self_loop", node_2="isolated")
    knowledge_graph = KnowledgeGraph.from_triplets(
        [isolated_triplet], scientific_domain="test_domain"
    )

    path = generator.generate_path_nodes(knowledge_graph, "isolated")
    assert path == ["isolated"]  # Should just have the start node

    # Test with invalid LLM response
    mock_response.content = "This response doesn't contain the expected format"

    # The generator should fall back to random selection when LLM response is invalid
    random.seed(42)  # Set a seed for reproducibility
    path = generator.generate_path_nodes(sample_knowledge_graph, "A")

    # Path should still be valid
    assert path[0] == "A"

    # Test the path edges formatting
    generator_internal = LLMWalkGenerator(max_steps=3, llm=mock_llm)
    path_edges = generator_internal._build_path_edges(
        ["A", "B", "C"], sample_knowledge_graph
    )

    # Should have edges for A->B and B->C
    assert len(path_edges) == 2
    assert path_edges[0][0] == "A"  # source
    assert path_edges[0][2] == "B"  # target
    assert path_edges[1][0] == "B"  # source
    assert path_edges[1][2] == "C"  # target

    # Test the path formatting for LLM
    path_str = generator_internal._format_path_for_llm(["A", "B", "C"], path_edges)

    # Should include all nodes and edges
    assert "A" in path_str
    assert "B" in path_str
    assert "C" in path_str

    # Test neighbors formatting
    neighbors = generator_internal._format_neighbors_for_llm(
        sample_knowledge_graph,
        "A",
        {"A"},  # A is visited
    )

    # Should have both unvisited and visited neighbors
    assert "unvisited" in neighbors
    assert "visited" in neighbors

    neighbors_str = generator_internal._format_neighbors_string(neighbors)

    # Should mention unvisited neighbors
    assert "Neighbors not yet visited" in neighbors_str

    # If A has neighbors in the sample graph, they should be mentioned
    successors = list(sample_knowledge_graph.get_successors("A"))
    predecessors = list(sample_knowledge_graph.get_predecessors("A"))

    if successors:
        assert any(successor in neighbors_str for successor in successors)
    if predecessors:
        assert any(predecessor in neighbors_str for predecessor in predecessors)
