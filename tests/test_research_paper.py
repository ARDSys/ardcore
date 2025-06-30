from pathlib import Path

import pytest

from ard.data.metadata import Metadata
from ard.data.research_paper import ResearchPaper


class TestResearchPaper:
    """Tests for the ResearchPaper class."""

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

    def test_save_and_get_pdf(self, paper):
        """Test saving and retrieving a PDF file."""
        # Sample PDF content
        pdf_content = b"%PDF-1.5\nThis is a fake PDF file for testing.\n%%EOF"

        # Save the PDF
        pdf_path = paper.save_pdf(pdf_content)
        assert Path(pdf_path).exists()

        # Get the PDF
        retrieved_content = paper.get_pdf()
        assert retrieved_content == pdf_content

    def test_save_and_get_pdf_with_custom_filename(self, paper):
        """Test saving and retrieving a PDF file with a custom filename."""
        # Sample PDF content
        pdf_content = b"%PDF-1.5\nThis is a fake PDF file for testing.\n%%EOF"
        custom_filename = "custom_name.pdf"

        # Save the PDF with a custom filename
        pdf_path = paper.save_pdf(pdf_content, filename=custom_filename)
        assert Path(pdf_path).exists()

        # Get the PDF with the custom filename
        retrieved_content = paper.get_pdf(filename=custom_filename)
        assert retrieved_content == pdf_content

    def test_save_and_get_extracted_text(self, paper):
        """Test saving and retrieving extracted text."""
        # Sample text content
        abstract = "This is a test abstract."
        introduction = "This is a test introduction."

        # Save the text sections
        abstract_path = paper.save_extracted_text(abstract, section="abstract")
        intro_path = paper.save_extracted_text(introduction, section="introduction")

        assert Path(abstract_path).exists()
        assert Path(intro_path).exists()

        # Get the text sections
        retrieved_abstract = paper.get_extracted_text(section="abstract")
        retrieved_intro = paper.get_extracted_text(section="introduction")

        assert retrieved_abstract == abstract
        assert retrieved_intro == introduction

    def test_save_and_get_full_text(self, paper):
        """Test saving and retrieving full text."""
        # Sample full text content
        full_text = "This is the full text of the paper."

        # Save the full text
        full_text_path = paper.save_extracted_text(full_text)
        assert Path(full_text_path).exists()

        # Get the full text
        retrieved_full_text = paper.get_extracted_text()
        assert retrieved_full_text == full_text

    def test_list_extracted_sections(self, paper):
        """Test listing extracted text sections."""
        # Save some text sections
        paper.save_extracted_text("Abstract content", section="abstract")
        paper.save_extracted_text("Introduction content", section="introduction")
        paper.save_extracted_text("Methods content", section="methods")
        paper.save_extracted_text("Full text content")  # This is the full text

        # List the sections
        sections = paper.list_extracted_sections()

        # Check that all sections are listed (except "full")
        assert len(sections) >= 3  # At least 3 sections
        assert "abstract" in sections or any(s.lower() == "abstract" for s in sections)
        assert "introduction" in sections or any(
            s.lower() == "introduction" for s in sections
        )
        assert "methods" in sections or any(s.lower() == "methods" for s in sections)
        assert "full" not in sections  # "full" should not be included in the list
