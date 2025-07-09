from unittest.mock import MagicMock, patch

import pytest

from ardcore.data.metadata import Metadata
from ardcore.data.research_paper import ResearchPaper


def test_dataset_item_creation():
    """Test that a DatasetItem can be created with a Metadata object."""
    metadata = Metadata(doi="10.1234/test.paper")
    item = ResearchPaper(metadata)

    assert item.get_metadata() == metadata
    assert item.get_metadata().doi == "10.1234/test.paper"


def test_from_metadata_factory():
    """Test the from_metadata factory method."""
    metadata = Metadata(
        authors=["John Doe", "Jane Smith"],
        title="Test Paper",
        doi="10.1234/test.paper",
        additional_metadata={"keywords": ["test", "example"]},
    )

    item = ResearchPaper.from_metadata(metadata)

    assert item.get_metadata() == metadata
    assert item.get_metadata().authors == ["John Doe", "Jane Smith"]
    assert item.get_metadata().title == "Test Paper"
    assert item.get_metadata().doi == "10.1234/test.paper"
    assert item.get_metadata().additional_metadata["keywords"] == ["test", "example"]


def test_from_doi_factory():
    """Test the from_doi factory method."""
    item = ResearchPaper.from_doi("10.5678/journal.test")

    assert item.get_metadata().doi == "10.5678/journal.test"
    assert item.get_metadata().pm_id is None


def test_from_pm_id_factory():
    """Test the from_pm_id factory method."""
    item = ResearchPaper.from_pm_id("12345678")

    assert item.get_metadata().pm_id == "12345678"
    assert item.get_metadata().doi is None


def test_metadata_privacy():
    """Test that the metadata attribute is private."""
    metadata = Metadata(pm_id="87654321")
    item = ResearchPaper(metadata)

    # Verify we can access metadata through the getter
    assert item.get_metadata() == metadata

    # Verify the attribute is private (starts with underscore)
    assert hasattr(item, "_metadata")
    assert not hasattr(item, "metadata")


@patch("ardcore.data.dataset_item.DatasetItem.list_kg_versions")
@patch("ardcore.data.triplets.Triplets.from_dataset_item")
def test_get_triplets_with_version(mock_from_dataset_item, mock_list_kg_versions):
    """Test getting triplets with a specified KG version."""
    # Setup
    metadata = Metadata(doi="10.1234/test.paper")
    item = ResearchPaper(metadata)
    mock_list_kg_versions.return_value = ["baseline_1", "baseline_2"]
    mock_triplets = MagicMock()
    mock_from_dataset_item.return_value = mock_triplets

    # Test with specified version
    result = item.get_triplets(kg_version="baseline_1")

    # Assertions
    assert result == mock_triplets
    mock_from_dataset_item.assert_called_once_with(
        item, "baseline_1", build_graph=False
    )


@patch("ardcore.data.dataset_item.DatasetItem.list_kg_versions")
@patch("ardcore.data.triplets.Triplets.from_dataset_item")
def test_get_triplets_with_build_graph(mock_from_dataset_item, mock_list_kg_versions):
    """Test getting triplets with build_graph parameter."""
    # Setup
    metadata = Metadata(doi="10.1234/test.paper")
    item = ResearchPaper(metadata)
    mock_list_kg_versions.return_value = ["baseline_1", "baseline_2"]
    mock_triplets = MagicMock()
    mock_from_dataset_item.return_value = mock_triplets

    # Test with build_graph=True
    result = item.get_triplets(kg_version="baseline_1", build_graph=True)

    # Assertions
    assert result == mock_triplets
    mock_from_dataset_item.assert_called_once_with(item, "baseline_1", build_graph=True)


@patch("ardcore.data.dataset_item.DatasetItem.list_kg_versions")
@patch("ardcore.data.triplets.Triplets.from_dataset_item")
def test_get_triplets_latest_version(mock_from_dataset_item, mock_list_kg_versions):
    """Test getting triplets with the latest KG version when none is specified."""
    # Setup
    metadata = Metadata(doi="10.1234/test.paper")
    item = ResearchPaper(metadata)
    mock_list_kg_versions.return_value = ["baseline_1", "baseline_2", "baseline_3"]
    mock_triplets = MagicMock()
    mock_from_dataset_item.return_value = mock_triplets

    # Test without specifying version (should use latest)
    result = item.get_triplets()

    # Assertions
    assert result == mock_triplets
    mock_from_dataset_item.assert_called_once_with(
        item, "baseline_3", build_graph=False
    )


@patch("ardcore.data.dataset_item.DatasetItem.list_kg_versions")
def test_get_triplets_no_versions(mock_list_kg_versions):
    """Test that get_triplets raises an error when no KG versions are available."""
    # Setup
    metadata = Metadata(doi="10.1234/test.paper")
    item = ResearchPaper(metadata)
    mock_list_kg_versions.return_value = []

    # Test with no versions available
    with pytest.raises(FileNotFoundError) as excinfo:
        item.get_triplets()

    assert f"No knowledge graph versions found for item {item.id}" in str(excinfo.value)


@patch("ardcore.data.dataset_item.DatasetItem.list_kg_versions")
def test_get_triplets_invalid_version(mock_list_kg_versions):
    """Test that get_triplets raises an error when an invalid KG version is specified."""
    # Setup
    metadata = Metadata(doi="10.1234/test.paper")
    item = ResearchPaper(metadata)
    mock_list_kg_versions.return_value = ["baseline_1", "baseline_2"]

    # Test with invalid version
    with pytest.raises(ValueError) as excinfo:
        item.get_triplets(kg_version="nonexistent_version")

    assert "Knowledge graph version 'nonexistent_version' not found" in str(
        excinfo.value
    )
