import os
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from ardcore.data.triplets import Triplet
from ardcore.knowledge_graph.knowledge_graph import KnowledgeGraph
from ardcore.utils.embedder import Embedder


@pytest.fixture
def sample_knowledge_graph():
    """Create a sample knowledge graph for testing."""
    # Create triplets for a simple graph
    triplets = [
        Triplet(node_1="A", edge="connects_to", node_2="B"),
        Triplet(node_1="B", edge="leads_to", node_2="C"),
        Triplet(node_1="C", edge="results_in", node_2="D"),
    ]

    return KnowledgeGraph.from_triplets(triplets, scientific_domain="test_domain")


@pytest.fixture
def mock_sentence_transformer():
    """Create a mock SentenceTransformer for testing."""
    with patch("sentence_transformers.SentenceTransformer") as mock_st:
        # Configure the mock to return fixed embeddings
        mock_model = MagicMock()

        def encode_side_effect(texts, **kwargs):
            # Create a mapping of text to embeddings
            embedding_map = {
                "A": [0.1, 0.2, 0.3],
                "B": [0.2, 0.3, 0.4],
                "C": [0.3, 0.4, 0.5],
                "D": [0.4, 0.5, 0.6],
            }

            # Handle both single text and list of texts
            if isinstance(texts, str):
                return np.array(embedding_map.get(texts, [0.5, 0.6, 0.7]))
            else:
                return np.array(
                    [embedding_map.get(text, [0.5, 0.6, 0.7]) for text in texts]
                )

        mock_model.encode.side_effect = encode_side_effect
        mock_st.return_value = mock_model
        yield mock_st


@pytest.fixture
def mock_gemini():
    """Create a mock Gemini client for testing."""
    with (
        patch.dict(os.environ, {"GOOGLE_API_KEY": "fake-api-key-for-testing"}),
        patch("google.genai.Client") as mock_gemini_client,
    ):
        mock_client = MagicMock()
        mock_client.models.embed_content.return_value = MagicMock(
            embeddings=[
                MagicMock(values=[0.1, 0.2, 0.3]),
                MagicMock(values=[0.2, 0.3, 0.4]),
                MagicMock(values=[0.3, 0.4, 0.5]),
                MagicMock(values=[0.4, 0.5, 0.6]),
            ]
        )
        mock_gemini_client.return_value = mock_client
        yield mock_gemini_client


def test_embedder_initialization():
    """Test that the Embedder initializes correctly."""
    embedder = Embedder(
        embedding_provider="sentence_transformer",
        model_name="test-model",
        cache_embeddings=True,
        distance_metric="cosine",
    )

    assert embedder.model_name == "test-model"
    assert embedder.cache_embeddings is True
    assert embedder.distance_metric == "cosine"
    assert embedder._model is None
    assert embedder._embeddings == {}


def test_embed(sample_knowledge_graph, mock_sentence_transformer):
    """Test calculating embeddings for a knowledge graph."""
    embedder = Embedder(
        embedding_provider="sentence_transformer",
        model_name="test-model",
        cache_embeddings=True,
        distance_metric="cosine",
    )

    # Calculate embeddings
    words = list(sample_knowledge_graph.get_nodes())
    embedding_vectors = embedder.embed(words)

    # Check that embeddings were calculated for all nodes
    assert len(embedding_vectors) == len(words)
    expected_nodes = {"A", "B", "C", "D"}
    assert len(words) == len(expected_nodes)  # Ensure words list matches expected

    for vector in embedding_vectors:
        assert isinstance(vector, np.ndarray)
        assert vector.shape == (3,)

    # Check that embeddings were cached
    # The cache _embeddings should still map words to their vectors
    assert set(embedder._embeddings.keys()) == expected_nodes
    assert embedder._embeddings["A"].shape == (3,)
    assert embedder._embeddings["B"].shape == (3,)
    assert embedder._embeddings["C"].shape == (3,)
    assert embedder._embeddings["D"].shape == (3,)


def test_embed_gemini(mock_gemini):
    """Test calculating embeddings for a knowledge graph using Gemini."""
    embedder = Embedder(
        embedding_provider="gemini",
        model_name="test-model",
        cache_embeddings=True,
        distance_metric="cosine",
        api_key="fake-api-key-for-testing",  # Provide fake API key for testing
    )

    # Calculate embeddings
    words = ["A", "B", "C", "D"]
    embedding_vectors = embedder.embed(words)

    # Check that embeddings were calculated for all nodes
    assert len(embedding_vectors) == len(words)
    for vector in embedding_vectors:
        assert isinstance(vector, np.ndarray)
        assert vector.shape == (3,)

    # Check that embeddings were cached
    # The cache _embeddings should still map words to their vectors
    assert set(embedder._embeddings.keys()) == set(words)
    assert embedder._embeddings["A"].shape == (3,)
    assert embedder._embeddings["B"].shape == (3,)
    assert embedder._embeddings["C"].shape == (3,)
    assert embedder._embeddings["D"].shape == (3,)

    # Mock should have been called only once
    assert mock_gemini.return_value.models.embed_content.call_count == 1


def test_get_embedding(mock_sentence_transformer):
    """Test getting an embedding for a single text."""
    embedder = Embedder(
        embedding_provider="sentence_transformer",
        model_name="test-model",
        cache_embeddings=True,
        distance_metric="cosine",
    )

    # Get embedding for a text
    embedding = embedder.get_embedding("A")

    # Check that the embedding was calculated
    assert embedding.shape == (3,)
    assert np.array_equal(embedding, np.array([0.1, 0.2, 0.3]))

    # Check that the embedding was cached
    assert "A" in embedder._embeddings

    # Get the same embedding again (should use cache)
    embedding2 = embedder.get_embedding("A")

    # Check that the embedding is the same
    assert np.array_equal(embedding, embedding2)

    # Mock should have been called only once
    assert mock_sentence_transformer.return_value.encode.call_count == 1


def test_get_embeddings(mock_sentence_transformer):
    """Test getting embeddings for multiple texts."""
    embedder = Embedder(
        embedding_provider="sentence_transformer",
        model_name="test-model",
        cache_embeddings=True,
        distance_metric="cosine",
    )

    # Get embeddings for multiple texts
    embeddings = embedder.get_embeddings(["A", "B", "C"])

    # Check that embeddings were calculated for all texts
    assert set(embeddings.keys()) == {"A", "B", "C"}
    assert np.array_equal(embeddings["A"], np.array([0.1, 0.2, 0.3]))
    assert np.array_equal(embeddings["B"], np.array([0.2, 0.3, 0.4]))
    assert np.array_equal(embeddings["C"], np.array([0.3, 0.4, 0.5]))

    # Check that embeddings were cached
    assert set(embedder._embeddings.keys()) == {"A", "B", "C"}


def test_calculate_distance(mock_sentence_transformer):
    """Test calculating distance between two texts."""
    embedder = Embedder(
        embedding_provider="sentence_transformer",
        model_name="test-model",
        distance_metric="cosine",
    )

    # Test with default metric (cosine)
    cosine_distance = embedder.calculate_distance("A", "B")
    assert 0 <= cosine_distance <= 1

    # Test with explicit cosine metric
    cosine_distance_explicit = embedder.calculate_distance("A", "B", metric="cosine")
    assert 0 <= cosine_distance_explicit <= 1
    assert cosine_distance == cosine_distance_explicit  # Should be the same as default

    # Test with euclidean metric (overriding default)
    euclidean_distance = embedder.calculate_distance("A", "B", metric="euclidean")
    assert euclidean_distance > 0

    # Test with dot product metric (overriding default)
    dot_distance = embedder.calculate_distance("A", "B", metric="dot")
    assert (
        dot_distance < 0
    )  # Dot product distance is negative because smaller is better

    # Test with invalid metric
    with pytest.raises(ValueError, match="Unknown distance metric: invalid"):
        embedder.calculate_distance("A", "B", metric="invalid")


def test_calculate_similarity(mock_sentence_transformer):
    """Test calculating similarity between two texts."""
    embedder = Embedder(
        embedding_provider="sentence_transformer",
        model_name="test-model",
        distance_metric="cosine",
    )

    # Test with default metric (cosine)
    cosine_similarity = embedder.calculate_similarity("A", "B")
    assert 0 <= cosine_similarity <= 1

    # Test with explicit cosine metric
    cosine_similarity_explicit = embedder.calculate_similarity(
        "A", "B", metric="cosine"
    )
    assert 0 <= cosine_similarity_explicit <= 1
    assert (
        cosine_similarity == cosine_similarity_explicit
    )  # Should be the same as default

    # Test with euclidean metric (overriding default)
    euclidean_similarity = embedder.calculate_similarity("A", "B", metric="euclidean")
    assert 0 <= euclidean_similarity <= 1

    # Test with dot product metric (overriding default)
    dot_similarity = embedder.calculate_similarity("A", "B", metric="dot")
    assert dot_similarity > 0

    # Test with invalid metric
    with pytest.raises(ValueError, match="Unknown distance metric: invalid"):
        embedder.calculate_similarity("A", "B", metric="invalid")


def test_save_and_load_embeddings():
    """Test saving and loading embeddings to/from a file."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_file = os.path.join(temp_dir, "embeddings.json")

        # Create an embedder and add some embeddings
        embedder = Embedder(
            embedding_provider="sentence_transformer",
            model_name="test-model",
            cache_embeddings=True,
            distance_metric="cosine",
        )
        embedder._embeddings = {
            "A": np.array([0.1, 0.2, 0.3]),
            "B": np.array([0.2, 0.3, 0.4]),
        }

        # Save embeddings to file
        embedder.save_to_file(temp_file)

        # Verify the file contains valid JSON
        with open(temp_file, "r", encoding="utf-8") as f:
            json_content = f.read()
            assert json_content  # File should not be empty
            import json

            data = json.loads(json_content)  # Should parse without errors
            assert "embeddings" in data

        # Create a new embedder and load embeddings from file
        embedder2 = Embedder(
            embedding_provider="sentence_transformer",
            model_name="test-model",
            cache_embeddings=True,
            distance_metric="cosine",
        )
        embedder2.load_from_file(temp_file)

        # Check that embeddings were loaded correctly
        assert set(embedder2._embeddings.keys()) == {"A", "B"}
        assert np.array_equal(embedder2._embeddings["A"], np.array([0.1, 0.2, 0.3]))
        assert np.array_equal(embedder2._embeddings["B"], np.array([0.2, 0.3, 0.4]))


def test_clear_cache():
    """Test clearing the embedding cache."""
    embedder = Embedder(
        embedding_provider="sentence_transformer",
        model_name="test-model",
        cache_embeddings=True,
        distance_metric="cosine",
    )
    embedder._embeddings = {
        "A": np.array([0.1, 0.2, 0.3]),
        "B": np.array([0.2, 0.3, 0.4]),
    }

    # Clear cache
    embedder.clear_cache()

    # Check that cache is empty
    assert embedder._embeddings == {}
