from ardcore.data.metadata import Metadata


def test_metadata_citations():
    """Test that a Metadata object can have citations."""
    # Create a metadata object with citations
    cited_paper1 = Metadata(doi="10.1234/cited.1", title="Cited Paper 1")
    cited_paper2 = Metadata(doi="10.1234/cited.2", title="Cited Paper 2")

    paper = Metadata(
        doi="10.1234/main.paper",
        title="Main Paper",
        citations=[cited_paper1, cited_paper2],
    )

    # Check that the citations are stored correctly
    assert len(paper.citations) == 2
    assert paper.citations[0].doi == "10.1234/cited.1"
    assert paper.citations[1].doi == "10.1234/cited.2"


def test_metadata_cited_by():
    """Test that a Metadata object can have cited_by references."""
    # Create a metadata object with cited_by references
    citing_paper1 = Metadata(doi="10.1234/citing.1", title="Citing Paper 1")
    citing_paper2 = Metadata(doi="10.1234/citing.2", title="Citing Paper 2")

    paper = Metadata(
        doi="10.1234/main.paper",
        title="Main Paper",
        cited_by=[citing_paper1, citing_paper2],
    )

    # Check that the cited_by references are stored correctly
    assert len(paper.cited_by) == 2
    assert paper.cited_by[0].doi == "10.1234/citing.1"
    assert paper.cited_by[1].doi == "10.1234/citing.2"


def test_metadata_serialization_with_citations():
    """Test serialization of Metadata with citations."""
    # Create a metadata object with citations
    cited_paper = Metadata(doi="10.1234/cited.paper", title="Cited Paper")
    citing_paper = Metadata(doi="10.1234/citing.paper", title="Citing Paper")

    paper = Metadata(
        doi="10.1234/main.paper",
        title="Main Paper",
        citations=[cited_paper],
        cited_by=[citing_paper],
    )

    # Convert to dictionary and check citations
    paper_dict = paper.to_dict()
    assert "citations" in paper_dict
    assert len(paper_dict["citations"]) == 1
    assert paper_dict["citations"][0]["doi"] == "10.1234/cited.paper"

    # Check cited_by
    assert "cited_by" in paper_dict
    assert len(paper_dict["cited_by"]) == 1
    assert paper_dict["cited_by"][0]["doi"] == "10.1234/citing.paper"

    # Convert to JSON and back
    json_str = paper.to_json()
    reconstructed = Metadata.from_json(json_str)

    # Check that citations are preserved
    assert len(reconstructed.citations) == 1
    assert reconstructed.citations[0].doi == "10.1234/cited.paper"
    assert len(reconstructed.cited_by) == 1
    assert reconstructed.cited_by[0].doi == "10.1234/citing.paper"
