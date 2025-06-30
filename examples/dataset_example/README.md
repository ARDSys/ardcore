# Dataset Example

This example demonstrates how to use the `Dataset` class from the ARD library to manage collections of research papers and other dataset items.

## Overview

The example shows:

1. How to create individual `DatasetItem` objects with metadata
2. How to combine them into a `Dataset`
3. How to load a `Dataset` directly from a directory
4. How to access and display information from the dataset

## Running the Example

To run this example:

```bash
# From the project root directory
python examples/dataset_example/example.py
```

## What to Expect

When you run the example, it will:

1. Create sample data in the `storage` directory
2. Create a dataset by manually adding individual items
3. Create a dataset by loading all items from a directory
4. Display information about the items in the dataset

## Code Explanation

The example demonstrates two main ways to create a Dataset:

1. **Creating a Dataset from individual items**:
   ```python
   # Create individual DatasetItem objects
   item1 = DatasetItem(metadata1)
   item2 = DatasetItem(metadata2)
   
   # Create a dataset with these items
   dataset = Dataset([item1, item2])
   ```

2. **Loading a Dataset from a directory**:
   ```python
   # Load a dataset from a directory containing DatasetItem objects
   dataset_from_dir = Dataset.from_local(storage_dir_path)
   ```

The example also shows how to access the items in a dataset and retrieve their metadata:

```python
for item in dataset.items:
    metadata = item.get_metadata()
    # Access metadata properties
    print(metadata.title)
    print(metadata.doi)
    print(metadata.authors)
```

## Implementation Notes

- The `Dataset` class is a simple container for `DatasetItem` objects
- The `from_local` method loads items from a directory using `DatasetItem.from_local`
- Each `DatasetItem` is identified by a unique ID, which is used for storage
- The example creates a fresh set of data each time it runs to avoid conflicts 