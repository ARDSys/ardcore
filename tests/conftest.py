# This file can contain shared fixtures for tests

# Add any shared fixtures here if needed

import os
import tempfile

import pytest

from ardcore.data.triplets import Triplet
from ardcore.knowledge_graph.knowledge_graph import KnowledgeGraph
from ardcore.storage.file import LocalStorageBackend, StorageManager


@pytest.fixture
def reset_storage_manager():
    """Reset the StorageManager singleton between tests."""
    # Reset the singleton instance
    StorageManager._instance = None
    StorageManager._backends = {}
    StorageManager._default_backend_name = "local"

    # Save original environment
    original_env = os.environ.copy()

    yield

    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)


@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


@pytest.fixture
def local_storage_backend(temp_dir):
    """Create a LocalStorageBackend instance for testing."""
    return LocalStorageBackend(temp_dir)


@pytest.fixture
def storage_manager_with_local_backend(reset_storage_manager, temp_dir):
    """Create a StorageManager with a local backend for testing."""
    backend = LocalStorageBackend(temp_dir)
    manager = StorageManager()
    manager.register_backend("test_local", backend)
    return manager, "test_local"


@pytest.fixture
def sample_knowledge_graph():
    """Create a sample knowledge graph for testing."""
    # Create triplets for a simple graph
    triplets = [
        # Main path: A -> B -> C -> D
        Triplet(node_1="A", edge="connects_to", node_2="B"),
        Triplet(node_1="B", edge="leads_to", node_2="C"),
        Triplet(node_1="C", edge="results_in", node_2="D"),
        # Alternative path: A -> E -> D
        Triplet(node_1="A", edge="relates_to", node_2="E"),
        Triplet(node_1="E", edge="influences", node_2="D"),
        # Neighbors of path nodes
        Triplet(node_1="B", edge="interacts_with", node_2="F"),
        Triplet(node_1="G", edge="regulates", node_2="C"),
        Triplet(node_1="H", edge="binds_to", node_2="A"),
    ]

    return KnowledgeGraph.from_triplets(triplets, scientific_domain="test_domain")
