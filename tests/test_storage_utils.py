import pytest

from ardcore.storage.file.utils import get_subgraph_file_name, get_subgraph_name


@pytest.mark.parametrize("subgraph_name", ["test_subgraph", "tests/subgraph_123"])
def test_get_subgraph_file_name(subgraph_name):
    # Test basic functionality
    assert get_subgraph_file_name(subgraph_name) == f"{subgraph_name}.subgraph.json"


@pytest.mark.parametrize(
    "subgraph_file_name,expected_subgraph_name",
    [
        ("path/to/test_subgraph.subgraph.json", "test_subgraph"),
        ("test_subgraph.subgraph.json", "test_subgraph"),
        ("path/to/test-subgraph_123.subgraph.json", "test-subgraph_123"),
        (
            "6.2_to_7.7_kg_m²_Macrophage_subpopulation.subgraph.json",
            "6.2_to_7.7_kg_m²_Macrophage_subpopulation",
        ),
    ],
)
def test_get_subgraph_name(subgraph_file_name, expected_subgraph_name):
    assert get_subgraph_name(subgraph_file_name) == expected_subgraph_name
