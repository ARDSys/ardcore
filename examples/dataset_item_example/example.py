import os
from pathlib import Path

from loguru import logger

from ardcore.data.dataset_item import DatasetItem


def main():
    # Read a local dataset item
    example_data_dir = Path(__file__).parent / "storage"

    os.environ["ARD_DATA_DIR"] = str(example_data_dir)

    # get example data id
    example_data_id = os.listdir(example_data_dir)[0]
    # read metadata.json
    logger.info(f"Example data id: {example_data_id}")
    # metadata_path = example_data_dir / example_data_id / "metadata.json"
    # with open(metadata_path, "r") as f:
    #     metadata = json.load(f)
    # print(metadata)
    # print(hashlib.md5(metadata["Title"].encode("utf-8")).hexdigest())
    # create dataset item
    dataset_item = DatasetItem.from_local(example_data_id)

    # Print the metadata
    logger.info(dataset_item.get_metadata())


if __name__ == "__main__":
    main()
