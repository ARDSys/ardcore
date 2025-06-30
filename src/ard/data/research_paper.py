import json
from pathlib import Path
from typing import BinaryIO, Dict, List, Optional, Union

import pandas as pd
from langfuse.decorators import observe
from loguru import logger

from ard.data.chunking import ChunkingProtocol, FixedChunking
from ard.data.dataset_item import DataCategory, DatasetItem
from ard.data.metadata import Metadata, MetadataType
from ard.data.triplets import Triplet, Triplets

# extract_refine_generator no longer needed - TripletsGenerationPipeline handles this automatically
from ard.data.triplets_extractor.pipeline import TripletsGenerationPipeline
from ard.data.triplets_extractor.utils import find_triplets_in_text


class ResearchPaper(DatasetItem):
    """
    Represents a scientific research paper in the dataset.
    Provides methods for accessing and managing paper-specific files.
    """

    def __init__(self, metadata: Metadata, storage_backend: Optional[str] = None):
        # Ensure the metadata type is set to PAPER
        if metadata.type != MetadataType.PAPER:
            metadata.type = MetadataType.PAPER

        super().__init__(metadata, storage_backend)

    def save_pdf(
        self, pdf_data: Union[bytes, BinaryIO], filename: Optional[str] = None
    ) -> str:
        """
        Save the PDF file for this research paper.

        Args:
            pdf_data: PDF content as bytes or file-like object
            filename: Optional filename to use (defaults to paper_id.pdf)

        Returns:
            str: The full path where the PDF was saved
        """
        if filename is None:
            filename = f"{self.id}.pdf"

        return self.save_file(filename, pdf_data, category=DataCategory.RAW.value)

    def get_pdf(self, filename: Optional[str] = None) -> bytes:
        """
        Retrieve the PDF file for this research paper.

        Args:
            filename: Optional filename to retrieve (defaults to paper_id.pdf)

        Returns:
            bytes: The PDF content
        """
        if filename is None:
            filename = f"{self.id}.pdf"

        return self.get_file(filename, category=DataCategory.RAW.value)

    def save_extracted_text(self, text: str, section: Optional[str] = None) -> str:
        """
        Save extracted text from the paper.

        Args:
            text: The extracted text content
            section: Optional section name (e.g., 'abstract', 'introduction')

        Returns:
            str: The full path where the text was saved
        """
        if section:
            filename = f"text/{section}.txt"
        else:
            filename = "text/full.txt"

        return self.save_file(
            filename, text.encode("utf-8"), category=DataCategory.PROCESSED.value
        )

    def get_extracted_text(self, section: Optional[str] = None) -> str:
        """
        Retrieve extracted text from the paper.

        Args:
            section: Optional section name (e.g., 'abstract', 'introduction')

        Returns:
            str: The extracted text content
        """
        if section:
            filename = f"text/{section}.txt"
        else:
            filename = "text/full.txt"

        return self.get_file(filename, category=DataCategory.PROCESSED.value).decode(
            "utf-8"
        )

    def list_extracted_sections(self) -> List[str]:
        """
        List all extracted text sections available for this paper.

        Returns:
            List[str]: List of section names
        """
        files = self.list_files(category=DataCategory.PROCESSED.value)
        sections = []

        for file_path in files:
            # Normalize path separators to forward slashes
            normalized_path = file_path.replace("\\", "/")
            if normalized_path.startswith("text/") and normalized_path.endswith(".txt"):
                section = Path(normalized_path).stem
                if section != "full":
                    sections.append(section)

        return sections

    def save_processed_data(self, data: Dict, name: str) -> str:
        """
        Save processed data as JSON.

        Args:
            data: Dictionary to save as JSON
            name: Name of the processed data file (without extension)

        Returns:
            str: The full path where the data was saved
        """
        filename = f"{name}.json"
        json_data = json.dumps(data, indent=2).encode("utf-8")
        return self.save_file(
            filename, json_data, category=DataCategory.PROCESSED.value
        )

    def get_processed_data(self, name: str | None = None) -> Dict:
        """
        Retrieve processed data from JSON.

        Args:
            name: Name of the processed data file (without extension)

        Returns:
            Dict: The loaded JSON data
        """
        if name is None:
            filename = f"{self.id[:14]}_processed.json"
        else:
            filename = f"{name}.json"
        json_data = self.get_file(filename, category=DataCategory.PROCESSED.value)
        return json.loads(json_data.decode("utf-8"))

    def list_processed_data(self) -> List[str]:
        """
        List all processed data files available for this paper.

        Returns:
            List[str]: List of processed data file names (without extension)
        """
        files = self.list_files(category=DataCategory.PROCESSED.value)
        data_files = []

        for file_path in files:
            if file_path.endswith(".json") and not file_path.startswith("text/"):
                data_files.append(Path(file_path).stem)

        return data_files

    @observe()
    def generate_kg(
        self,
        config: Optional[Dict] = None,
        kg_version: Optional[str] = None,
        overwrite: bool = False,
    ) -> Triplets:
        """Generate triplets from research paper content using the standalone pipeline.

        This method is now a thin wrapper around TripletsGenerationPipeline, handling only
        dataset-specific concerns (caching, text retrieval, persistence, versioning).

        Args:
            config: Optional configuration dictionary (must include extraction_type)
            kg_version: Optional name of the KG version to use (e.g., 'baseline_1'). If not provided,
                       a new version will be created. If provided and the version exists with both
                       triplets.csv and config.json, processing will be skipped.
            overwrite: Whether to overwrite existing KG version

        Returns:
            Triplets: Generated triplets with metadata and configuration
        """
        logger.info(f"Starting KG pipeline for paper {self.id}")
        logger.info(f"KG version: {kg_version}")

        try:
            # Ensure config is not None
            if config is None:
                config = {}

            # Step 1: Handle caching - check if KG version already exists
            if kg_version is not None and not overwrite:
                try:
                    triplets = self.get_triplets(kg_version)
                    logger.info(
                        f"KG version {kg_version} already exists with required files, skipping processing"
                    )
                    return triplets
                except Exception:
                    pass  # KG version doesn't exist, continue with generation

            # Step 2: Get text content from storage (dataset-specific logic)
            try:
                # Try to get processed text data - this is dataset-specific
                data = self.get_processed_data()
                # Extract text from the processed data structure
                if "full_text" in data:
                    text = data["full_text"]
                elif "text" in data:
                    text = data["text"]
                else:
                    raise ValueError(
                        "No text content (either `full_text` or `text`) found in processed data"
                    )

                logger.debug(f"Retrieved text content ({len(text)} characters)")

            except Exception as e:
                logger.error(f"Failed to retrieve text content: {e}")
                raise

            # Step 3: Prepare metadata for pipeline
            item_metadata = self.get_metadata().to_dict()
            item_metadata["source"] = self.id  # Use paper ID as source

            # Step 4: Use standalone pipeline for triplet generation
            pipeline = TripletsGenerationPipeline(config=config)
            triplets = pipeline.generate_triplets(text=text, metadata=item_metadata)

            logger.info(f"Pipeline generated {len(triplets.triplets)} triplets")

            # Step 5: Handle versioning and persistence (dataset-specific logic)
            if kg_version is None:
                baseline_num = 1 + len(self.list_kg_versions())
                kg_version = f"baseline_{baseline_num}"

            logger.debug(f"Using kg_version: {kg_version}")

            # Save triplets to dataset storage
            triplets.save_to_dataset_item(
                self.id,
                kg_version,
                ResearchPaper,
                self._storage_backend_name,
            )
            logger.info(f"Saved all triplets to {kg_version} dir")
            logger.info("KG pipeline completed successfully")

            return triplets

        except Exception as e:
            logger.error(f"Error running KG pipeline: {e}")
            raise

    def add_context_to_triplets(self, kg_version: str) -> Triplets:
        """Add context to the triplets."""
        triplets = self.get_triplets(kg_version)
        config = triplets.config

        chunker = FixedChunking(
            chunk_size=config["chunk_size"], chunk_overlap=config["chunk_overlap"]
        )
        chunks = self.chunk_text(chunker)
        logger.debug(f"Created {len(chunks)} chunks for processing")

        all_triplets = []

        for i, chunk in enumerate(chunks):
            logger.debug(f"Adding context to chunk {i + 1}/{len(chunks)}")
            local_triplets = [
                triplet
                for triplet in triplets.triplets
                if triplet.metadata.get("chunk_id") == chunk["chunk_id"]
            ]
            local_triplets_with_snippet = [
                triplet
                for triplet in local_triplets
                if triplet.metadata.get("snippet") is not None
            ]
            local_triplets_without_snippet = [
                triplet
                for triplet in local_triplets
                if triplet.metadata.get("snippet") is None
            ]
            logger.debug(
                f"Found {len(local_triplets)} triplets for chunk {chunk['chunk_id']}"
            )
            new_triplets = find_triplets_in_text(
                chunk["chunk"],
                local_triplets_without_snippet,
                config.get("context_model", "gpt-4o"),
            )
            new_triplets = [
                Triplet(
                    t["node_1"],
                    t["edge"],
                    t["node_2"],
                    {"chunk_id": chunk["chunk_id"], "snippet": t["snippet"]},
                )
                for t in new_triplets
            ]
            local_triplets = local_triplets_with_snippet + new_triplets
            logger.debug(
                f"Found {len(local_triplets)} triplets for chunk {chunk['chunk_id']}"
            )
            all_triplets.extend(local_triplets)

        triplets = Triplets(
            all_triplets, config=config, item_metadata=self.get_metadata().to_dict()
        )

        triplets.save_to_dataset_item(
            self.id,
            kg_version,
            ResearchPaper,
            self._storage_backend_name,
        )

        return triplets

    @staticmethod
    def save_triplets_as_csv(triplets, output_path, backend):
        """
        Save triplets as a CSV file.

        Args:
            triplets (list): List of triplets
            output_path (str): Path to save the CSV file
        """
        # Create a DataFrame from the triplets
        df = pd.DataFrame(triplets)

        # Save the DataFrame as a CSV file
        df.to_csv(output_path, index=False)
        backend.save_file(
            output_path,
            df.to_csv(index=False).encode("utf-8"),
            category=DataCategory.PROCESSED.value,
        )

    def chunk_text(self, chunker: ChunkingProtocol) -> List[Dict[str, str]]:
        """
        Chunk text data using different methods.

        Args:
            method (Literal['fixed', 'sections', 'full']): Chunking method to use
            **kwargs: Additional arguments
                clean (bool): Whether to use cleaned text (default True)
                chunk_size (int): Size of chunks for fixed method (default 800)
                chunk_overlap (int): Overlap between chunks for fixed method (default 0)

        Returns:
            Union[List[str], str]: Chunked text as list or full text as string
        """
        logger.info(f"Chunking text using method: {chunker}")
        data = self.get_processed_data()
        text_chunks = chunker.chunk(data)
        chunks = self._parse_chunks(text_chunks)
        return chunks

    def _parse_chunks(self, chunks: List[Dict[str, str]]) -> List[Dict[str, str]]:
        return [
            {"chunk": chunk, "chunk_id": f"{self.id}_{i}"}
            for i, chunk in enumerate(chunks)
        ]
