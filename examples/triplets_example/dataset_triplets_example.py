#!/usr/bin/env python
"""
Example script demonstrating how to use the get_triplets method in the Dataset class.

This script shows how to:
1. Create a Dataset from a directory of DatasetItems
2. Get triplets from all items in the dataset
3. Handle errors when retrieving triplets
"""

import warnings
from pathlib import Path

from loguru import logger

from ard.data.dataset import Dataset


def main():
    """Run the dataset triplets example."""
    # Read a local dataset item
    example_data_dir = Path(__file__).parent / "storage"

    logger.info(f"Loading dataset from: {example_data_dir}")

    # Create a Dataset from the directory
    dataset = Dataset.from_local(example_data_dir)

    logger.info(f"Dataset loaded with {len(dataset)} items")

    # Get triplets from all items in the dataset
    logger.info("\n=== Getting Triplets from All Items ===")

    # First, try with skip_errors=True (default)
    logger.info("\nGetting triplets with skip_errors=True:")
    triplets_dict = dataset.get_triplets(build_graph=False)

    logger.info(f"Successfully retrieved triplets from {len(triplets_dict)} items")

    # Print information about each item's triplets
    for item_id, triplets in triplets_dict.items():
        logger.info(f"\nItem ID: {item_id}")
        logger.info(f"Number of triplets: {len(triplets)}")

        # Get nodes and edges without building the graph
        with warnings.catch_warnings(record=True) as w:
            nodes = triplets.get_nodes()
            edges = triplets.get_edges()
            if w:
                logger.warning(f"Warning: {w[0].message}")

        logger.info(f"Number of nodes: {len(nodes)}")
        logger.info(f"Number of edge types: {len(edges)}")
        logger.info(f"Edge types: {', '.join(sorted(edges))}")

    # Try with a specific KG version
    logger.info("\n=== Getting Triplets for a Specific KG Version ===")

    # Get all available KG versions for the first item
    if dataset.items:
        first_item = dataset.items[0]
        kg_versions = first_item.list_kg_versions()

        if kg_versions:
            logger.info(
                f"\nAvailable KG versions for item {first_item.id}: {kg_versions}"
            )

            # Get triplets for a specific version
            specific_version = kg_versions[0]
            logger.info(f"\nGetting triplets for version: {specific_version}")

            try:
                triplets_dict = dataset.get_triplets(
                    kg_version=specific_version, build_graph=True
                )
                logger.info(
                    f"Successfully retrieved triplets from {len(triplets_dict)} items"
                )
            except FileNotFoundError as e:
                logger.error(f"Error: {e}")
        else:
            logger.info(f"No KG versions found for item {first_item.id}")
    else:
        logger.info("No items found in the dataset")


if __name__ == "__main__":
    main()
