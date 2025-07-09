#!/usr/bin/env python3
"""
Example script demonstrating how to use the ARD storage system.
"""

import os
import sys
from pathlib import Path

from loguru import logger

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ardcore.data.metadata import Metadata
from ardcore.data.research_paper import ResearchPaper
from ardcore.storage.file import StorageManager


def main():
    # Set up a custom data directory for this example
    example_data_dir = Path(__file__).parent / "example_data"
    os.environ["ARD_DATA_DIR"] = str(example_data_dir)

    # Get the storage manager and register a local backend
    StorageManager()

    # Create a research paper
    metadata = Metadata(
        doi="10.1234/example.2023.001",
        title="Example Research Paper",
        authors=["John Doe", "Jane Smith"],
    )
    paper = ResearchPaper(metadata)

    logger.info(f"Created paper with ID: {paper.id}")

    # Save some example files
    sample_pdf = (
        b"%PDF-1.5\nThis is a fake PDF file for demonstration purposes only.\n%%EOF"
    )
    pdf_path = paper.save_pdf(sample_pdf)
    logger.info(f"Saved PDF to: {pdf_path}")

    # Save some extracted text
    abstract = "This is an example abstract for demonstration purposes."
    abstract_path = paper.save_extracted_text(abstract, section="abstract")
    logger.info(f"Saved abstract to: {abstract_path}")

    introduction = "This is an example introduction for demonstration purposes."
    intro_path = paper.save_extracted_text(introduction, section="introduction")
    logger.info(f"Saved introduction to: {intro_path}")

    # List all files for this paper
    logger.info("\nAll files for this paper:")
    all_files = paper.list_files()
    for file_path in all_files:
        logger.info(f"- {file_path}")

    # List extracted sections
    logger.info("\nExtracted sections:")
    sections = paper.list_extracted_sections()
    for section in sections:
        logger.info(f"- {section}")

    # Retrieve and print the abstract
    retrieved_abstract = paper.get_extracted_text(section="abstract")
    logger.info(f"\nRetrieved abstract: {retrieved_abstract}")

    # Delete a file
    paper.delete_file("text/introduction.txt", category="extracted")
    logger.info("\nDeleted introduction.txt")

    # List files again to confirm deletion
    logger.info("\nFiles after deletion:")
    all_files = paper.list_files()
    for file_path in all_files:
        logger.info(f"- {file_path}")


if __name__ == "__main__":
    main()
