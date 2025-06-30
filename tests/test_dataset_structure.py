import json
from pathlib import Path

import pytest

from ard.data.dataset_item import DataCategory
from ard.data.metadata import Metadata
from ard.data.research_paper import ResearchPaper


class ConcreteDatasetItem(ResearchPaper):
    """Concrete implementation of DatasetItem for testing."""

    pass


class TestDatasetStructure:
    """Tests for the structured file organization in DatasetItem."""

    @pytest.fixture
    def dataset_item(self, storage_manager_with_local_backend):
        """Create a DatasetItem instance with a local storage backend."""
        manager, backend_name = storage_manager_with_local_backend

        # Create a dataset item that uses this backend
        metadata = Metadata(doi="10.1234/test.item")
        return ConcreteDatasetItem(metadata, storage_backend=backend_name)

    def test_directory_structure_initialization(
        self, dataset_item, storage_manager_with_local_backend
    ):
        """Test that the standard directory structure is created on initialization."""
        manager, backend_name = storage_manager_with_local_backend
        backend = manager.get_backend(backend_name)

        # Check that standard directories exist
        for category in DataCategory:
            dir_path = backend._get_item_dir(dataset_item.id, category.value)
            assert dir_path.exists()
            assert dir_path.is_dir()

        # Check that metadata.json exists at the top level
        metadata_path = backend._get_item_dir(dataset_item.id) / "metadata.json"
        assert metadata_path.exists()
        assert metadata_path.is_file()

    def test_kg_version_management(self, dataset_item):
        """Test creating and listing KG versions."""
        # Create some KG versions
        version1 = dataset_item.create_kg_version("baseline_1")  # noqa: F841
        version2 = dataset_item.create_kg_version("improved_2")  # noqa: F841

        # List KG versions
        versions = dataset_item.list_kg_versions()
        assert "baseline_1" in versions
        assert "improved_2" in versions

        # Save some files in the KG versions
        dataset_item.save_kg_file(
            "baseline_1", "triplets.csv", b"subject,predicate,object"
        )
        dataset_item.save_kg_file("baseline_1", "graph.gpickle", b"binary data")
        dataset_item.save_kg_file("improved_2", "triplets.csv", b"better triplets")

        # List files in a specific version
        baseline_files = dataset_item.list_kg_files("baseline_1")
        assert "triplets.csv" in baseline_files
        assert "graph.gpickle" in baseline_files
        assert "config.json" in baseline_files

        # Get a file from a specific version
        triplets = dataset_item.get_kg_file("baseline_1", "triplets.csv")
        assert triplets == b"subject,predicate,object"

        # Try to create a version with an invalid name
        with pytest.raises(ValueError):
            dataset_item.create_kg_version("invalid/name")

    def test_invalid_category(self, dataset_item):
        """Test that using an invalid category raises an error."""
        with pytest.raises(ValueError):
            dataset_item.save_file("test.txt", b"test", category="invalid")

        # But a KG subdirectory should work
        dataset_item.save_file(
            "test.txt", b"test", category=f"{DataCategory.KG.value}/custom"
        )


class TestResearchPaperStructure:
    """Tests for the structured file organization in ResearchPaper."""

    @pytest.fixture
    def paper(self, storage_manager_with_local_backend):
        """Create a ResearchPaper instance with a local storage backend."""
        manager, backend_name = storage_manager_with_local_backend

        # Create a research paper that uses this backend
        metadata = Metadata(
            doi="10.1234/test.paper",
            title="Test Research Paper",
            authors=["John Doe", "Jane Smith"],
        )
        return ResearchPaper(metadata, storage_backend=backend_name)

    def test_pdf_storage(self, paper, storage_manager_with_local_backend):
        """Test that PDFs are stored in the raw directory."""
        manager, backend_name = storage_manager_with_local_backend
        backend = manager.get_backend(backend_name)

        # Save a PDF
        pdf_content = b"%PDF-1.5\nThis is a fake PDF file for testing.\n%%EOF"
        pdf_path = paper.save_pdf(pdf_content)

        # Check that it's in the raw directory
        raw_dir = backend._get_item_dir(paper.id, DataCategory.RAW.value)
        assert Path(pdf_path).parent == raw_dir

        # Get the PDF and verify content
        retrieved_pdf = paper.get_pdf()
        assert retrieved_pdf == pdf_content

    def test_extracted_text_storage(self, paper, storage_manager_with_local_backend):
        """Test that extracted text is stored in the processed directory."""
        manager, backend_name = storage_manager_with_local_backend
        backend = manager.get_backend(backend_name)

        # Save some extracted text
        abstract = "This is the abstract."
        abstract_path = paper.save_extracted_text(abstract, section="abstract")

        # Check that it's in the processed directory
        processed_dir = backend._get_item_dir(paper.id, DataCategory.PROCESSED.value)
        assert Path(abstract_path).parent.parent == processed_dir

        # Get the text and verify content
        retrieved_abstract = paper.get_extracted_text(section="abstract")
        assert retrieved_abstract == abstract

    def test_processed_data_storage(self, paper):
        """Test saving and retrieving processed data."""
        # Save some processed data
        relevance_data = {"score": 0.85, "keywords": ["science", "research"]}
        relevance_path = paper.save_processed_data(relevance_data, "relevance")  # noqa: F841

        summary_data = {"summary": "This is a summary.", "length": 120}
        summary_path = paper.save_processed_data(summary_data, "summary")  # noqa: F841

        # List processed data files
        data_files = paper.list_processed_data()
        assert "relevance" in data_files
        assert "summary" in data_files

        # Get processed data and verify content
        retrieved_relevance = paper.get_processed_data("relevance")
        assert retrieved_relevance == relevance_data

        retrieved_summary = paper.get_processed_data("summary")
        assert retrieved_summary == summary_data

    def test_kg_integration(self, paper):
        """Test knowledge graph integration with research papers."""
        # Create a KG version
        kg_version = paper.create_kg_version("paper_kg_v1")  # noqa: F841

        # Save some KG files
        paper.save_kg_file("paper_kg_v1", "triplets.csv", b"subject,predicate,object")
        paper.save_kg_file(
            "paper_kg_v1",
            "entities.json",
            json.dumps({"entities": ["entity1", "entity2"]}).encode("utf-8"),
        )

        # List KG files
        kg_files = paper.list_kg_files("paper_kg_v1")
        assert "triplets.csv" in kg_files
        assert "entities.json" in kg_files

        # Get KG files
        triplets = paper.get_kg_file("paper_kg_v1", "triplets.csv")
        assert triplets == b"subject,predicate,object"

        entities = json.loads(
            paper.get_kg_file("paper_kg_v1", "entities.json").decode("utf-8")
        )
        assert entities["entities"] == ["entity1", "entity2"]
