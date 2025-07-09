#!/usr/bin/env python
"""
Example script demonstrating how to use the Triplets class.

This script shows how to:
1. Load triplets from a dataset item
2. Analyze the triplets and graph structure
3. Extract a subgraph
4. Save triplets to a new file
"""

import os
import sys
from pathlib import Path

from loguru import logger

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ardcore.data.dataset_item import DatasetItem
from ardcore.data.triplets import Triplets


def main():
    """Run the triplets example."""
    # Read a local dataset item
    example_data_dir = Path(__file__).parent / "storage"

    os.environ["ARD_DATA_DIR"] = str(example_data_dir)

    # Example dataset item ID
    item_id = "2ee6f627fd89f026f5d5108731c7908ec5ee30db24a4002adc26e50756b47cf5"
    kg_version = "baseline_1"

    logger.info(f"Loading triplets for dataset item: {item_id}")
    logger.info(f"Knowledge graph version: {kg_version}")

    # Load triplets from the dataset item
    triplets = Triplets.from_dataset_item(item_id, kg_version)
    logger.info(triplets)

    # Get triplets for the dataset item
    item = DatasetItem.from_local(item_id)
    triplets = item.get_triplets(kg_version)
    logger.info(triplets)


if __name__ == "__main__":
    main()
