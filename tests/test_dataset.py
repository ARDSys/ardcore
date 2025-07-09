from unittest.mock import MagicMock, patch

import pytest

from ardcore.data.dataset import Dataset
from ardcore.data.dataset_item import DatasetItem
from ardcore.data.metadata import Metadata
from ardcore.data.research_paper import ResearchPaper
from ardcore.storage.file import StorageManager


def test_dataset_init():
    """Test that a Dataset can be created with a list of DatasetItems."""
    # Create some test DatasetItems
    item1 = ResearchPaper(Metadata(doi="10.1234/test.1"))
    item2 = ResearchPaper(Metadata(doi="10.1234/test.2"))
    items = [item1, item2]

    # Create a Dataset with these items
    dataset = Dataset(items)

    # Verify the items are stored correctly
    assert dataset.items == items
    assert len(dataset.items) == 2
    assert dataset.items[0].get_metadata().doi == "10.1234/test.1"
    assert dataset.items[1].get_metadata().doi == "10.1234/test.2"


def test_dataset_init_empty():
    """Test that a Dataset can be created with an empty list."""
    dataset = Dataset([])
    assert dataset.items == []
    assert len(dataset.items) == 0


@patch("os.listdir")
@patch("ardcore.data.dataset_item.DatasetItem.from_local")
def test_from_local(mock_from_local, mock_listdir, tmp_path):
    """Test the from_local method to create a Dataset from a local directory."""
    # Setup mock returns
    mock_listdir.return_value = ["item1", "item2", "item3"]

    # Create mock DatasetItems that will be returned by from_local
    mock_item1 = MagicMock(spec=DatasetItem)
    mock_item2 = MagicMock(spec=DatasetItem)
    mock_item3 = MagicMock(spec=DatasetItem)

    # Configure the mock to return different items for different calls
    mock_from_local.side_effect = [mock_item1, mock_item2, mock_item3]

    # Call the method under test using a temporary directory path
    test_path = str(tmp_path)
    storage_manager = MagicMock(spec=StorageManager)
    dataset = Dataset([]).from_local(
        test_path,
        daset_item_type=ResearchPaper,
        storage_manager=storage_manager,
        overwrite=False,
        kg_version="abc123",
    )

    # Verify the method called os.listdir with the correct path
    mock_listdir.assert_called_once_with(test_path)

    # Verify DatasetItem.from_local was called for each item
    assert mock_from_local.call_count == 3
    mock_from_local.assert_any_call("item1")
    mock_from_local.assert_any_call("item2")
    mock_from_local.assert_any_call("item3")

    # Verify the returned dataset has the correct items
    assert isinstance(dataset, Dataset)
    assert len(dataset.items) == 3
    assert dataset.items[0] in [mock_item1, mock_item2, mock_item3]
    assert dataset.items[1] in [mock_item1, mock_item2, mock_item3]
    assert dataset.items[2] in [mock_item1, mock_item2, mock_item3]


@patch("os.listdir")
@patch("ardcore.data.dataset_item.DatasetItem.from_local")
def test_from_local_empty_directory(mock_from_local, mock_listdir, tmp_path):
    """Test the from_local method with an empty directory."""
    # Setup mock to return an empty list
    mock_listdir.return_value = []

    # Call the method under test using a temporary directory path
    test_path = str(tmp_path)
    storage_manager = MagicMock(spec=StorageManager)
    dataset = Dataset([]).from_local(
        test_path,
        daset_item_type=ResearchPaper,
        storage_manager=storage_manager,
        overwrite=False,
        kg_version="abc123",
    )

    # Verify the method called os.listdir with the correct path
    mock_listdir.assert_called_once_with(test_path)

    # Verify DatasetItem.from_local was not called
    mock_from_local.assert_not_called()

    # Verify the returned dataset has no items
    assert isinstance(dataset, Dataset)
    assert len(dataset.items) == 0


@patch("os.listdir")
@patch("ardcore.data.dataset_item.DatasetItem.from_local")
@patch("loguru.logger.warning")
def test_from_local_error_handling(
    mock_logger_warning, mock_from_local, mock_listdir, tmp_path
):
    """Test error handling in the from_local method."""
    # Setup mock returns
    mock_listdir.return_value = ["item1", "item2", "error_item"]

    # Configure the mock to raise an exception for the third item
    mock_item1 = MagicMock(spec=DatasetItem)
    mock_item2 = MagicMock(spec=DatasetItem)
    mock_from_local.side_effect = [
        mock_item1,
        mock_item2,
        FileNotFoundError("Test error"),
    ]

    # Call the method under test using a temporary directory path
    test_path = str(tmp_path)
    storage_manager = MagicMock(spec=StorageManager)
    dataset = Dataset([]).from_local(
        test_path,
        daset_item_type=ResearchPaper,
        storage_manager=storage_manager,
        overwrite=False,
        kg_version="abc123",
    )

    # Verify the method still returns a dataset with the successful items
    assert isinstance(dataset, Dataset)
    assert len(dataset.items) == 2
    assert dataset.items[0] in [mock_item1, mock_item2]
    assert dataset.items[1] in [mock_item1, mock_item2]

    # Verify that the warning was logged
    mock_logger_warning.assert_called_once_with("Item error_item not found, skipping")


def test_repr():
    """Test the __repr__ method."""
    # Create a simple dataset with a controlled string representation
    metadata = Metadata(doi="10.1234/test.repr")
    item = ResearchPaper(metadata)
    dataset = Dataset([item])

    # Verify __repr__ contains the expected format
    result = repr(dataset)
    assert "Dataset(items=[" in result
    assert "DatasetItem" in result
    assert "])" in result


def test_str():
    """Test the __str__ method."""
    # Create a simple dataset with a controlled string representation
    metadata = Metadata(doi="10.1234/test.str")
    item = ResearchPaper(metadata)
    dataset = Dataset([item])

    # Verify __str__ contains the expected format
    result = str(dataset)
    assert "Dataset(items=[" in result
    assert "DatasetItem" in result
    assert "])" in result


def test_len():
    """Test the __len__ method."""
    # Create Datasets with different numbers of items
    dataset_empty = Dataset([])
    dataset_one = Dataset([MagicMock(spec=DatasetItem)])
    dataset_multiple = Dataset(
        [MagicMock(spec=DatasetItem), MagicMock(spec=DatasetItem)]
    )

    # Verify __len__ returns the correct counts
    assert len(dataset_empty) == 0
    assert len(dataset_one) == 1
    assert len(dataset_multiple) == 2


@patch("os.listdir")
@patch("ardcore.data.dataset_item.DatasetItem.from_local")
def test_dataset_from_local(mock_from_local, mock_listdir, tmp_path):
    """Test that a Dataset can be created from a local directory."""
    # Mock the os.listdir function to return a list of item IDs
    mock_listdir.return_value = ["item1", "item2", "item3"]

    # Mock the DatasetItem.from_local method to return a DatasetItem for each item ID
    # except for item3, which will raise a FileNotFoundError
    def mock_from_local_side_effect(item_id):
        if item_id == "item3":
            raise FileNotFoundError(f"Item {item_id} not found")
        return MagicMock(spec=DatasetItem)

    mock_from_local.side_effect = mock_from_local_side_effect

    # Create a Dataset from a local directory using a temporary directory path
    test_path = str(tmp_path)
    storage_manager = MagicMock(spec=StorageManager)
    dataset = Dataset.from_local(
        test_path,
        daset_item_type=ResearchPaper,
        storage_manager=storage_manager,
        overwrite=False,
        kg_version="abc123",
    )

    # Verify the items are loaded correctly
    assert len(dataset.items) == 2  # item3 should be skipped
    mock_listdir.assert_called_once_with(test_path)
    assert mock_from_local.call_count == 3


def test_dataset_get_triplets():
    """Test that triplets can be retrieved from all items in a dataset."""
    # Create some test DatasetItems with mocked id property and get_triplets method
    item1 = MagicMock(spec=DatasetItem)
    item1.id = "item1"
    mock_triplets1 = MagicMock()
    item1.get_triplets.return_value = mock_triplets1

    item2 = MagicMock(spec=DatasetItem)
    item2.id = "item2"
    mock_triplets2 = MagicMock()
    item2.get_triplets.return_value = mock_triplets2

    items = [item1, item2]

    # Create a Dataset with these items
    dataset = Dataset(items)

    # Get triplets from all items in the dataset
    triplets_dict = dataset.get_triplets(kg_version="baseline_1", build_graph=True)

    # Verify the triplets are retrieved correctly
    assert len(triplets_dict) == 2
    assert triplets_dict["item1"] == mock_triplets1
    assert triplets_dict["item2"] == mock_triplets2

    # Verify the get_triplets method was called with the correct arguments
    item1.get_triplets.assert_called_once_with(
        kg_version="baseline_1", build_graph=True
    )
    item2.get_triplets.assert_called_once_with(
        kg_version="baseline_1", build_graph=True
    )


def test_dataset_get_triplets_with_errors():
    """Test that errors are handled correctly when retrieving triplets."""
    # Create some test DatasetItems with mocked id property and get_triplets method
    item1 = MagicMock(spec=DatasetItem)
    item1.id = "item1"
    mock_triplets = MagicMock()
    item1.get_triplets.return_value = mock_triplets

    item2 = MagicMock(spec=DatasetItem)
    item2.id = "item2"
    item2.get_triplets.side_effect = FileNotFoundError("No KG versions found")

    items = [item1, item2]

    # Create a Dataset with these items
    dataset = Dataset(items)

    # Get triplets from all items in the dataset with skip_errors=True
    triplets_dict = dataset.get_triplets(skip_errors=True)

    # Verify only the successful item is included
    assert len(triplets_dict) == 1
    assert triplets_dict["item1"] == mock_triplets
    assert "item2" not in triplets_dict

    # Verify the get_triplets method was called on both items
    item1.get_triplets.assert_called_once()
    item2.get_triplets.assert_called_once()

    # Create a new dataset with the same items for testing skip_errors=False
    dataset = Dataset(items)

    # Reset the mocks
    item1.get_triplets.reset_mock()
    item2.get_triplets.reset_mock()

    # Get triplets from all items in the dataset with skip_errors=False
    with pytest.raises(FileNotFoundError):
        dataset.get_triplets(skip_errors=False)
