import json

import pytest

from ard.data.metadata import Metadata, MetadataType


def test_metadata_creation_with_doi():
    """Test creating metadata with a DOI."""
    metadata = Metadata(
        authors=["John Doe", "Jane Smith"],
        title="Example Paper Title",
        doi="10.1234/example.doi",
    )

    assert metadata.authors == ["John Doe", "Jane Smith"]
    assert metadata.title == "Example Paper Title"
    assert metadata.doi == "10.1234/example.doi"
    assert metadata.pm_id is None
    assert metadata.additional_metadata == {}


def test_metadata_creation_with_pm_id():
    """Test creating metadata with a PMID."""
    metadata = Metadata(
        title="Another Example Paper",
        pm_id="12345678",
        additional_metadata={"keywords": ["test", "example"]},
    )

    assert metadata.authors is None
    assert metadata.title == "Another Example Paper"
    assert metadata.doi is None
    assert metadata.pm_id == "12345678"
    assert metadata.additional_metadata["keywords"] == ["test", "example"]


def test_metadata_creation_with_both_identifiers():
    """Test creating metadata with both DOI and PMID."""
    metadata = Metadata(
        authors=["Researcher One", "Researcher Two"],
        title="Comprehensive Study",
        doi="10.5678/comprehensive.study",
        pm_id="87654321",
    )

    assert metadata.authors == ["Researcher One", "Researcher Two"]
    assert metadata.title == "Comprehensive Study"
    assert metadata.doi == "10.5678/comprehensive.study"
    assert metadata.pm_id == "87654321"


def test_metadata_creation_without_identifiers():
    """Test that creating metadata without DOI or PMID raises an error."""
    with pytest.raises(ValueError) as excinfo:
        Metadata(authors=["Author Name"], title="Paper Without Identifiers")

    assert "At least one identifier" in str(excinfo.value)
    assert "must be provided" in str(excinfo.value)


def test_metadata_equality_with_matching_doi():
    """Test equality when both objects have the same DOI."""
    metadata1 = Metadata(doi="10.1234/paper.123")
    metadata2 = Metadata(doi="10.1234/paper.123")

    assert metadata1 == metadata2


def test_metadata_equality_with_matching_pm_id():
    """Test equality when both objects have the same PMID."""
    metadata1 = Metadata(pm_id="12345678")
    metadata2 = Metadata(pm_id="12345678")

    assert metadata1 == metadata2


def test_metadata_equality_with_different_doi():
    """Test equality when both objects have different DOIs."""
    metadata1 = Metadata(doi="10.1234/paper.123")
    metadata2 = Metadata(doi="10.1234/paper.456")

    assert metadata1 != metadata2


def test_metadata_equality_with_different_pm_id():
    """Test equality when both objects have different PMIDs."""
    metadata1 = Metadata(pm_id="12345678")
    metadata2 = Metadata(pm_id="87654321")

    assert metadata1 != metadata2


def test_metadata_equality_with_different_identifier_types():
    """Test equality when objects have different identifier types."""
    metadata1 = Metadata(doi="10.1234/paper.123")
    metadata2 = Metadata(pm_id="12345678")

    assert metadata1 != metadata2


def test_metadata_equality_with_both_identifiers_matching():
    """Test equality when both objects have matching DOI and PMID."""
    metadata1 = Metadata(doi="10.1234/paper.123", pm_id="12345678")
    metadata2 = Metadata(doi="10.1234/paper.123", pm_id="12345678")

    assert metadata1 == metadata2


def test_metadata_equality_with_both_identifiers_not_matching():
    """Test equality when both objects have non-matching DOI and PMID."""
    metadata1 = Metadata(doi="10.1234/paper.123", pm_id="12345678")
    metadata2 = Metadata(doi="10.5678/paper.456", pm_id="87654321")

    assert metadata1 != metadata2


def test_metadata_equality_with_inconsistent_identifiers():
    """Test that comparing metadata with inconsistent identifiers raises an error."""
    metadata1 = Metadata(doi="10.1234/paper.123", pm_id="12345678")
    metadata2 = Metadata(doi="10.1234/paper.123", pm_id="87654321")

    with pytest.raises(ValueError) as excinfo:
        metadata1 == metadata2

    assert "Inconsistent identifiers" in str(excinfo.value)
    assert "pm_id, id don't match" in str(excinfo.value)


def test_metadata_equality_with_non_metadata_object():
    """Test equality with a non-Metadata object."""
    metadata = Metadata(doi="10.1234/paper.123")

    assert metadata != "not a metadata object"
    assert metadata != 12345
    assert metadata != {"doi": "10.1234/paper.123"}


def test_metadata_id_consistency():
    """Test that the id property returns consistent values for the same metadata."""
    metadata1 = Metadata(doi="10.1234/paper.123", pm_id="12345678")
    metadata2 = Metadata(doi="10.1234/paper.123", pm_id="12345678")

    # Same metadata should produce the same ID
    assert metadata1.id == metadata2.id

    # ID should be consistent when called multiple times
    assert metadata1.id == metadata1.id


def test_metadata_id_uniqueness():
    """Test that different metadata produces different IDs."""
    metadata1 = Metadata(doi="10.1234/paper.123", pm_id="12345678")
    metadata2 = Metadata(doi="10.5678/paper.456", pm_id="87654321")
    metadata3 = Metadata(
        doi="10.1234/paper.123", pm_id="87654321"
    )  # Same DOI, different PMID
    metadata4 = Metadata(
        doi="10.5678/paper.456", pm_id="12345678"
    )  # Different DOI, same PMID

    # Different metadata should produce different IDs
    assert metadata1.id != metadata2.id
    assert metadata1.id != metadata3.id
    assert metadata1.id != metadata4.id
    assert metadata2.id != metadata3.id
    assert metadata2.id != metadata4.id
    assert metadata3.id != metadata4.id


def test_metadata_id_with_doi_only():
    """Test that the id property works correctly with only DOI."""
    metadata = Metadata(doi="10.1234/paper.123")

    # ID should be a non-empty string
    assert isinstance(metadata.id, str)
    assert len(metadata.id) > 0

    # Same DOI should produce the same ID
    metadata2 = Metadata(doi="10.1234/paper.123")
    assert metadata.id == metadata2.id

    # Different DOI should produce different ID
    metadata3 = Metadata(doi="10.5678/paper.456")
    assert metadata.id != metadata3.id


def test_metadata_id_with_pm_id_only():
    """Test that the id property works correctly with only PMID."""
    metadata = Metadata(pm_id="12345678")

    # ID should be a non-empty string
    assert isinstance(metadata.id, str)
    assert len(metadata.id) > 0

    # Same PMID should produce the same ID
    metadata2 = Metadata(pm_id="12345678")
    assert metadata.id == metadata2.id

    # Different PMID should produce different ID
    metadata3 = Metadata(pm_id="87654321")
    assert metadata.id != metadata3.id


def test_metadata_id_format():
    """Test the format of the generated ID."""
    metadata = Metadata(doi="10.1234/paper.123", pm_id="12345678")

    # ID should be a hexadecimal string (SHA-256 produces 64 hex characters)
    assert isinstance(metadata.id, str)
    assert len(metadata.id) == 64
    assert all(c in "0123456789abcdef" for c in metadata.id)


def test_metadata_type_enum():
    """Test the MetadataType enum."""
    assert MetadataType.PAPER.value == "paper"


def test_metadata_with_type():
    """Test that a Metadata object can be created with a type."""
    metadata = Metadata(doi="10.1234/test.paper", type=MetadataType.PAPER)
    assert metadata.type == MetadataType.PAPER


def test_metadata_default_type():
    """Test that a Metadata object has a default type of PAPER."""
    metadata = Metadata(doi="10.1234/test.paper")
    assert metadata.type == MetadataType.PAPER


def test_metadata_with_doi():
    """Test that a Metadata object can be created with a DOI."""
    metadata = Metadata(doi="10.1186/s12974-020-01774-9")
    assert metadata.doi == "10.1186/s12974-020-01774-9"
    assert metadata.type == MetadataType.PAPER
    assert metadata.id is not None


def test_metadata_with_pm_id():
    """Test that a Metadata object can be created with a PMID."""
    metadata = Metadata(pm_id=32238175)
    assert metadata.pm_id == 32238175
    assert metadata.type == MetadataType.PAPER
    assert metadata.id is not None


def test_metadata_with_multiple_ids():
    """Test that a Metadata object can be created with multiple IDs."""
    metadata = Metadata(
        doi="10.1186/s12974-020-01774-9",
        pm_id=32238175,
        pmc_id="PMC7115095",
        gs_id="11636663754596445832",
    )
    assert metadata.doi == "10.1186/s12974-020-01774-9"
    assert metadata.pm_id == 32238175
    assert metadata.pmc_id == "PMC7115095"
    assert metadata.gs_id == "11636663754596445832"
    assert metadata.id is not None


def test_metadata_validation():
    """Test that a Metadata object requires at least one identifier."""
    with pytest.raises(ValueError):
        Metadata(title="Test Paper")


def test_metadata_id_generation():
    """Test that the ID is generated consistently."""
    metadata1 = Metadata(doi="10.1186/s12974-020-01774-9")
    metadata2 = Metadata(doi="10.1186/s12974-020-01774-9")
    assert metadata1.id == metadata2.id


def test_metadata_to_dict():
    """Test converting metadata to a dictionary."""
    metadata = Metadata(
        doi="10.1186/s12974-020-01774-9",
        pm_id=32238175,
        pmc_id="PMC7115095",
        gs_id="11636663754596445832",
        title="Transcriptomic profiling of microglia and astrocytes throughout aging.",
        abstract="Test abstract",
        category="CITES_HALD",
        type=MetadataType.PAPER,
        additional_metadata={"keywords": ["test", "example"]},
    )

    data = metadata.to_dict()
    assert data["doi"] == "10.1186/s12974-020-01774-9"
    assert data["pm_id"] == 32238175
    assert data["pmc_id"] == "PMC7115095"
    assert data["gs_id"] == "11636663754596445832"
    assert (
        data["title"]
        == "Transcriptomic profiling of microglia and astrocytes throughout aging."
    )
    assert data["abstract"] == "Test abstract"
    assert data["category"] == "CITES_HALD"
    assert data["_internal"]["type"] == "paper"
    assert data["_internal"]["additional_metadata"]["keywords"] == ["test", "example"]


def test_metadata_to_json():
    """Test converting metadata to JSON."""
    metadata = Metadata(
        doi="10.1186/s12974-020-01774-9",
        pm_id=32238175,
        title="Test Paper",
        type=MetadataType.PAPER,
    )

    json_str = metadata.to_json()
    data = json.loads(json_str)
    assert data["doi"] == "10.1186/s12974-020-01774-9"
    assert data["pm_id"] == 32238175
    assert data["title"] == "Test Paper"
    assert "id" in data


def test_metadata_from_dict():
    """Test creating metadata from a dictionary."""
    data = {
        "DOI": "10.1186/s12974-020-01774-9",
        "PM_ID": 32238175,
        "PMC_ID": "PMC7115095",
        "GS_ID": "11636663754596445832",
        "Title": "Transcriptomic profiling of microglia and astrocytes throughout aging.",
        "Abstract": "Test abstract",
        "Category": "CITES_HALD",
        "_internal": {
            "type": "paper",
            "additional_metadata": {"keywords": ["test", "example"]},
        },
    }

    metadata = Metadata.from_dict(data)
    assert metadata.doi == "10.1186/s12974-020-01774-9"
    assert metadata.pm_id == 32238175
    assert metadata.pmc_id == "PMC7115095"
    assert metadata.gs_id == "11636663754596445832"
    assert (
        metadata.title
        == "Transcriptomic profiling of microglia and astrocytes throughout aging."
    )
    assert metadata.abstract == "Test abstract"
    assert metadata.category == "CITES_HALD"
    assert metadata.type == MetadataType.PAPER
    assert metadata.additional_metadata["keywords"] == ["test", "example"]


def test_metadata_from_json():
    """Test creating metadata from JSON."""
    json_str = """
    {
        "DOI": "10.1186/s12974-020-01774-9",
        "PM_ID": 32238175,
        "PMC_ID": "PMC7115095",
        "GS_ID": "11636663754596445832",
        "Title": "Transcriptomic profiling of microglia and astrocytes throughout aging.",
        "Abstract": "Test abstract",
        "Category": "CITES_HALD",
        "_internal": {
            "type": "paper",
            "additional_metadata": {"keywords": ["test", "example"]}
        }
    }
    """

    metadata = Metadata.from_json(json_str)
    assert metadata.doi == "10.1186/s12974-020-01774-9"
    assert metadata.pm_id == 32238175
    assert metadata.pmc_id == "PMC7115095"
    assert metadata.gs_id == "11636663754596445832"
    assert (
        metadata.title
        == "Transcriptomic profiling of microglia and astrocytes throughout aging."
    )
    assert metadata.abstract == "Test abstract"
    assert metadata.category == "CITES_HALD"
    assert metadata.type == MetadataType.PAPER
    assert metadata.additional_metadata["keywords"] == ["test", "example"]


def test_metadata_from_example():
    """Test creating metadata from the example JSON."""
    json_str = """
    {
      "EXTRA_ID": null,
      "PM_ID": 32238175,
      "GS_ID": "11636663754596445832",
      "PMC_ID": "PMC7115095",
      "DOI": "10.1186/s12974-020-01774-9",
      "Category": "CITES_HALD",
      "Title": "Transcriptomic profiling of microglia and astrocytes throughout aging.",
      "Abstract": "Activation of microglia and astrocytes, a prominent hallmark of both aging and Alzheimer's disease (AD), has been suggested to contribute to aging and AD progression, but the underlying cellular and molecular mechanisms are largely unknown. We performed RNA-seq analyses on microglia and astrocytes freshly isolated from wild-type and APP-PS1 (AD) mouse brains at five time points to elucidate their age-related gene-expression profiles. Our results showed that from 4\\u2009months onward, a set of age-related genes in microglia and astrocytes exhibited consistent upregulation or downregulation (termed \\"age-up\\"/\\"age-down\\" genes) relative to their expression at the young-adult stage (2\\u2009months). And most age-up genes were more highly expressed in AD mice at the same time points. Bioinformatic analyses revealed that the age-up genes in microglia were associated with the inflammatory response, whereas these genes in astrocytes included widely recognized AD risk genes, genes associated with synaptic transmission or elimination, and peptidase-inhibitor genes. Overall, our RNA-seq data provide a valuable resource for future investigations into the roles of microglia and astrocytes in aging- and amyloid-\\u03b2-induced AD pathologies.",
      "GS_Meta": "[{'position': 1, 'title': 'Transcriptomic profiling of microglia and astrocytes throughout aging', 'result_id': 'iG5i9we8faEJ', 'link': 'https://link.springer.com/article/10.1186/S12974-020-01774-9', 'snippet': 'Activation of microglia and astrocytes, a prominent hallmark of both aging and Alzheimer\\u2019s disease (AD), has been suggested to contribute to aging and AD progression, but the underlying cellular and molecular mechanisms are largely unknown.\\\\nWe performed RNA-seq analyses on microglia and astrocytes freshly isolated from wild-type and APP-PS1 (AD) mouse brains at five time points to elucidate their age-related gene-expression profiles.\\\\nOur results showed that from 4 months onward, a set of age-related genes in microglia and astrocytes exhibited consistent upregulation or downregulation (termed \\u201cage-up\\u201d/\\u201cage-down\\u201d genes) relative to their expression at the young-adult stage (2 months). And most age-up genes were more highly expressed in AD mice at the same time points. Bioinformatic analyses revealed that the age-up genes in microglia were associated with the inflammatory response, whereas these genes in astrocytes included widely recognized AD risk genes, genes associated with synaptic transmission or elimination, and peptidase-inhibitor genes.\\\\nOverall, our RNA-seq data provide a valuable resource for future investigations into the roles of microglia and astrocytes in aging- and amyloid-\\u03b2-induced AD pathologies.', 'publication_info': {'summary': 'J Pan, N Ma, B Yu, W Zhang, J Wan - Journal of Neuroinflammation, 2020 - Springer'}, 'resources': [{'title': 'springer.com', 'file_format': 'PDF', 'link': 'https://link.springer.com/content/pdf/10.1186/s12974-020-01774-9.pdf'}], 'inline_links': {'serpapi_cite_link': 'https://serpapi.com/search.json?engine=google_scholar_cite&hl=en&q=iG5i9we8faEJ', 'cited_by': {'total': 169, 'link': 'https://scholar.google.com/scholar?cites=11636663754596445832&as_sdt=5,33&sciodt=0,33&hl=en', 'cites_id': '11636663754596445832', 'serpapi_scholar_link': 'https://serpapi.com/search.json?as_sdt=5%2C33&cites=11636663754596445832&engine=google_scholar&hl=en'}, 'related_pages_link': 'https://scholar.google.com/scholar?q=related:iG5i9we8faEJ:scholar.google.com/&scioq=10.1186/s12974-020-01774-9&hl=en&as_sdt=0,33', 'serpapi_related_pages_link': 'https://serpapi.com/search.json?as_sdt=0%2C33&engine=google_scholar&hl=en&num=10&q=related%3AiG5i9we8faEJ%3Ascholar.google.com%2F&scisbd=0&start=0', 'versions': {'total': 13, 'link': 'https://scholar.google.com/scholar?cluster=11636663754596445832&hl=en&as_sdt=0,33', 'cluster_id': '11636663754596445832', 'serpapi_scholar_link': 'https://serpapi.com/search.json?as_sdt=0%2C33&cluster=11636663754596445832&engine=google_scholar&hl=en'}}}]",
      "Scraped_Date": "2025-02-01 20:19:00.508210",
      "ID": "2ee6f627fd89f026f5d5108731c7908ec5ee30db24a4002adc26e50756b47cf5"
    }
    """

    metadata = Metadata.from_json(json_str)
    assert metadata.doi == "10.1186/s12974-020-01774-9"
    assert metadata.pm_id == 32238175
    assert metadata.pmc_id == "PMC7115095"
    assert metadata.gs_id == "11636663754596445832"
    assert metadata.extra_id is None
    assert metadata.category == "CITES_HALD"
    assert (
        metadata.title
        == "Transcriptomic profiling of microglia and astrocytes throughout aging."
    )
    assert "Activation of microglia and astrocytes" in metadata.abstract
    assert metadata.gs_meta is not None
    assert metadata.scraped_date == "2025-02-01 20:19:00.508210"
    assert (
        metadata.id
        == "2ee6f627fd89f026f5d5108731c7908ec5ee30db24a4002adc26e50756b47cf5"
    )
    assert metadata.type == MetadataType.PAPER
