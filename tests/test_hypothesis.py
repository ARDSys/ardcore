import json

import pytest

from ard.hypothesis.hypothesis import Hypothesis, HypothesisGeneratorProtocol
from ard.hypothesis.saver import JSONParser, MarkdownParser
from ard.subgraph import Subgraph
from ard.subgraph.subgraph_generator import ShortestPathGenerator


@pytest.fixture
def sample_subgraph(sample_knowledge_graph):
    """Create a sample subgraph for testing."""
    subgraph = Subgraph.from_two_nodes(
        sample_knowledge_graph, "A", "D", ShortestPathGenerator()
    )
    subgraph._context = "This is a test context for the subgraph."
    return subgraph


class DummyGenerator(HypothesisGeneratorProtocol):
    """A dummy generator for testing purposes."""

    def run(self, subgraph):
        """Returns a dummy hypothesis statement."""
        return Hypothesis(
            title="Test Hypothesis",
            statement="This is a test hypothesis",
            source=subgraph,
            metadata={"key": "value"},
            method=self,
            references=["Reference 1", "Reference 2"],
        )

    def __str__(self):
        return "DummyGenerator"

    def to_json(self):
        return {"name": "DummyGenerator", "params": {}}


@pytest.fixture
def sample_hypothesis(sample_subgraph):
    """Create a sample hypothesis for testing."""
    method = DummyGenerator()
    hypothesis = Hypothesis.from_subgraph(
        sample_subgraph,
        method,
    )
    return hypothesis


def test_hypothesis_from_subgraph(sample_subgraph):
    """Test creating a hypothesis from a subgraph."""
    method = DummyGenerator()

    # Create hypothesis from subgraph
    hypothesis = Hypothesis.from_subgraph(sample_subgraph, method)

    # Check the generated hypothesis
    assert hypothesis is not None
    assert hypothesis.source == sample_subgraph
    assert hypothesis.method == method
    assert hypothesis.statement == "This is a test hypothesis"


def test_json_parser(sample_hypothesis):
    """Test the JSONParser with a hypothesis."""
    # Create parser
    parser = JSONParser()

    # Parse hypothesis
    json_str = parser.parse(sample_hypothesis)

    # Verify it's valid JSON
    json_data = json.loads(json_str)

    # Check contents
    assert json_data["title"] == "Test Hypothesis"
    assert json_data["text"] == "This is a test hypothesis"
    assert "source" in json_data
    assert json_data["metadata"] == {"key": "value"}
    assert json_data["method_name"] == "DummyGenerator"
    assert "method" in json_data
    assert json_data["method"]["name"] == "DummyGenerator"
    assert json_data["references"] == ["Reference 1", "Reference 2"]


def test_markdown_parser(sample_subgraph, sample_hypothesis):
    """Test the MarkdownParser with a hypothesis."""
    # Create parser
    parser = MarkdownParser()

    # Parse hypothesis
    markdown = parser.parse(sample_hypothesis)

    # Check markdown structure
    assert (
        f"""
# Test Hypothesis

This is a test hypothesis

## References
### 1. Reference 1
### 2. Reference 2

## Context
This is a test context for the subgraph.

## Subgraph
```cypher
{sample_subgraph.to_cypher_string()}
```
"""
        == markdown
    )
