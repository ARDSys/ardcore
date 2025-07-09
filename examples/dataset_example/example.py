import os
import shutil
from pathlib import Path

from loguru import logger

from ardcore.data.dataset import Dataset
from ardcore.data.dataset_item import DatasetItem
from ardcore.data.metadata import Metadata


def setup_example_environment():
    """Set up the example environment with sample data."""
    # Define the storage directory
    storage_dir = Path(__file__).parent / "storage"

    # Clear existing data to avoid duplicates
    if storage_dir.exists():
        shutil.rmtree(storage_dir)
    storage_dir.mkdir(exist_ok=True)

    # Set the environment variable for storage
    os.environ["ARD_DATA_DIR"] = str(storage_dir)

    # Create sample items with different metadata
    items_data = [
        Metadata(
            doi="10.1234/example.1",
            title="Example Paper 1",
            authors=["Author A", "Author B"],
        ),
        Metadata(
            doi="10.1234/example.2", title="Example Paper 2", authors=["Author C"]
        ),
        Metadata(
            doi="10.1234/example.3",
            title="Example Paper 3",
            authors=["Author A", "Author D"],
        ),
    ]

    # Create the dataset items
    created_items = []
    for i, metadata in enumerate(items_data):
        # Create a dataset item
        item = DatasetItem(metadata)
        created_items.append(item)
        logger.info(f"Created sample item {i + 1} with ID: {item.id}")
        logger.info(f"  Title: {metadata.title}")
        logger.info(f"  DOI: {metadata.doi}")
        logger.info(f"  Authors: {', '.join(metadata.authors)}")

    return storage_dir, created_items


def display_dataset_info(dataset, title):
    """Display information about a dataset."""
    logger.info(f"\n--- {title} ---")
    logger.info(f"Dataset contains {len(dataset)} items")

    for i, item in enumerate(dataset.items):
        metadata = item.get_metadata()
        logger.info(f"\nItem {i + 1}:")
        logger.info(f"  ID: {item.id}")
        logger.info(f"  Title: {metadata.title}")
        logger.info(f"  DOI: {metadata.doi}")
        if metadata.authors:
            logger.info(f"  Authors: {', '.join(metadata.authors)}")


def main():
    """Demonstrate the usage of the Dataset class."""
    # Set up the example environment
    logger.info("Setting up example environment...")
    storage_dir, created_items = setup_example_environment()

    # Create a dataset with the first two items
    dataset = Dataset(created_items[:2])
    display_dataset_info(dataset, "Dataset created from individual items")
    logger.info(f"Dataset representation: {dataset}")

    # Load a dataset from a directory
    dataset_from_dir = Dataset.from_local(str(storage_dir))
    display_dataset_info(dataset_from_dir, "Dataset loaded from directory")


if __name__ == "__main__":
    main()
