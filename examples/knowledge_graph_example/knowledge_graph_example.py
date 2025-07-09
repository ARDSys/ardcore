from pathlib import Path

from loguru import logger

from ardcore.data.dataset import Dataset
from ardcore.knowledge_graph import KnowledgeGraph


def main():
    # Read a local dataset item
    example_data_dir = Path(__file__).parent / "storage"

    # Create a Dataset from the directory
    dataset = Dataset.from_local(example_data_dir)

    # First, try with skip_errors=True (default)
    logger.info("Getting triplets with skip_errors=True:")
    triplets_dict = dataset.get_triplets(build_graph=False)
    triplets_list = list(triplets_dict.values())

    # Create a KnowledgeGraph from triplets
    kg = KnowledgeGraph.from_triplets(triplets_list)
    logger.info("Original KnowledgeGraph:")
    logger.info(kg)


if __name__ == "__main__":
    main()
