"""
Standalone pipeline for generating triplets from text.

This module provides a clean, reusable interface for triplet generation
that is independent of dataset storage infrastructure.

Concept Clarification:
- **Triplets**: Raw extracted relationships (subject, predicate, object) from text
- **Knowledge Graph**: Structured graph built from collections of triplets
- This pipeline generates triplets; knowledge graphs are built from triplets
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

from loguru import logger

from ardcore.data.chunking import FixedChunking
from ardcore.data.triplets import Triplet, Triplets
from ardcore.data.triplets_extractor.config.factory import ExtractorFactory
from ardcore.data.triplets_extractor.config.pipeline import PipelineConfig
from ardcore.data.triplets_extractor.utils import (
    find_triplets_in_text,
    generate_response,
)
from ardcore.data.triplets_refiner import refine_triplets_with_llm
from ardcore.data.types import KGGenerator


class TripletsGenerationPipeline:
    """
    Standalone pipeline for generating triplets from text.

    Simple interface: text string → Triplets object
    Completely independent of DatasetItem/storage infrastructure.

    Example usage:
        # Constructor config approach
        pipeline = TripletsGenerationPipeline(config={
            "chunk_size": 1000,
            "max_workers": 8,
            "extraction_type": "refine",
            "extractor_model_name": "gpt-4o"
        })

        triplets = pipeline.generate_triplets(
            text="Research paper content...",
            metadata={"source": "paper_123"},
            config_overrides={"max_workers": 4}
        )
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the triplets generation pipeline with configuration.

        Args:
            config: Optional configuration dictionary. If not provided, defaults will be used.
                   See PipelineConfig and extractor configs for available parameters.
        """
        if config is None:
            config = {}

        # Create pipeline configuration for common pipeline parameters
        self.pipeline_config = PipelineConfig.from_dict(config)

        # Store full config for extractor factory
        self.full_config = config.copy()

        logger.info(
            f"Initialized TripletsGenerationPipeline with pipeline config: {self.pipeline_config}"
        )

    def generate_triplets(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        config_overrides: Optional[Dict[str, Any]] = None,
    ) -> Triplets:
        """
        Generate triplets from input text.

        Args:
            text: Input text to extract triplets from
            metadata: Optional metadata about the source (e.g., {"source": "paper_123"})
            config_overrides: Optional configuration overrides for this specific run

        Returns:
            Triplets: Generated triplets with metadata and configuration

        Raises:
            ValueError: If text is empty or None
            Exception: If generation pipeline fails
        """
        if not text or not isinstance(text, str):
            raise ValueError("Text input must be a non-empty string")

        if metadata is None:
            metadata = {}

        # Merge configuration with any overrides for this run
        effective_config = self.full_config.copy()
        if config_overrides:
            effective_config.update(config_overrides)

        # Update pipeline config with overrides
        effective_pipeline_config = self.pipeline_config.merge_with(config_overrides)

        logger.info(f"Starting triplets generation for text of {len(text)} characters")
        logger.debug(f"Effective config: {effective_config}")

        try:
            # Create generator using factory
            kg_generator = ExtractorFactory.create_generator(effective_config)

            # Generate chunk ID prefix from metadata
            chunk_id_prefix = metadata.get("source", "item")

            # Step 1: Chunk the text
            chunks = self.chunk_text(
                text, effective_pipeline_config, chunk_id_prefix=chunk_id_prefix
            )
            logger.debug(f"Created {len(chunks)} chunks for processing")

            # Step 2: Process chunks in parallel
            all_triplets = self._process_chunks_parallel(
                chunks, kg_generator, effective_pipeline_config
            )

            # Step 3: Convert to Triplet objects
            triplet_list = [
                Triplet(
                    t["node_1"],
                    t["edge"],
                    t["node_2"],
                    {"chunk_id": t["chunk_id"], "snippet": t["snippet"]},
                )
                for t in all_triplets
            ]

            # Step 4: Refine triplets with LLM
            logger.debug(f"Refining {len(triplet_list)} triplets")
            refined_triplets = refine_triplets_with_llm(
                Triplets(
                    triplet_list,
                    config=effective_config,
                    item_metadata=metadata,
                ),
                effective_pipeline_config.scientific_domain,
            )
            logger.debug(f"Refined {len(refined_triplets.triplets)} triplets")

            # Step 5: Clean refined triplets (remove metadata for final output)
            refined_triplet_list = [
                Triplet(
                    t.node_1,
                    t.edge,
                    t.node_2,
                    {},
                )
                for t in refined_triplets.triplets
            ]

            # Step 6: Create final Triplets object with complete config
            # Merge pipeline config and effective config for complete configuration record
            complete_config = effective_pipeline_config.to_dict()
            complete_config.update(effective_config)

            logger.debug(f"Generated {len(refined_triplet_list)} final triplets")
            triplets = Triplets(
                refined_triplet_list,
                config=complete_config,
                item_metadata=metadata,
            )

            logger.info("Triplets generation completed successfully")
            return triplets

        except Exception as e:
            logger.error(f"Error in triplets generation pipeline: {e}")
            raise

    def get_config(self) -> PipelineConfig:
        """
        Get the current pipeline configuration.

        Returns:
            PipelineConfig: Current pipeline configuration
        """
        return self.pipeline_config

    def update_config(self, new_config: Dict[str, Any]) -> None:
        """
        Update the pipeline configuration.

        Args:
            new_config: New configuration parameters to merge with current config
        """
        self.pipeline_config = self.pipeline_config.merge_with(new_config)
        self.full_config.update(new_config)
        logger.info(f"Updated pipeline config: {self.pipeline_config}")

    def __str__(self) -> str:
        """String representation of the pipeline."""
        return f"TripletsGenerationPipeline(config={self.pipeline_config})"

    def __repr__(self) -> str:
        """Detailed string representation of the pipeline."""
        return self.__str__()

    # Text Processing Components

    @staticmethod
    def distill_text(content: str, model_name: str, **kwargs) -> str:
        """
        Distill text content using LLM to create scientific summaries.

        Args:
            content: Input text to distill
            model_name: Name of the LLM model to use
            **kwargs: Additional arguments (unused, for compatibility)

        Returns:
            str: Distilled text content (summary + bullet points)

        Raises:
            Exception: If LLM generation fails
        """
        logger.debug(f"Distilling text using model {model_name}")
        sys_msg1 = (
            "You respond with a concise scientific summary, including reasoning. "
            "You never use names or references. "
        )
        usr_prompt1 = (
            "In a matter-of-fact voice, rewrite this '{text}'. "
            "The writing must stand on its own and provide all background needed, "
            "and include details. Do not include names, figures, plots or "
            "citations in your response, only facts."
        )

        usr_prompt2 = (
            "Provide a bullet point list of the key facts and reasoning in {summary}. "
            "The writing must stand on its own and provide all background needed, and "
            "include details. Do not include figures, plots or citations in your response. Think step by step. "
        )

        try:
            summary = generate_response(
                model_name, sys_msg1, usr_prompt1.format(text=content)
            )
            logger.debug("Generated summary")
            bullet_list = generate_response(
                model_name, sys_msg1, usr_prompt2.format(summary=summary)
            )
            logger.debug("Generated bullet list")
            return summary + "\n\n" + bullet_list
        except Exception as e:
            logger.error(f"Error in text distillation: {e}")
            raise

    def chunk_text(
        self, text: str, config: PipelineConfig, chunk_id_prefix: str = "chunk"
    ) -> List[Dict[str, str]]:
        """
        Chunk input text into manageable pieces.

        Args:
            text: Input text to chunk
            config: Pipeline configuration
            chunk_id_prefix: Prefix for chunk IDs (default: "chunk")

        Returns:
            List[Dict[str, str]]: List of chunks with 'chunk' and 'chunk_id' keys
        """
        logger.info(
            f"Chunking text of {len(text)} characters using {config.chunk_method}"
        )

        # Create chunker based on configuration
        if config.chunk_method == "fixed":
            chunker = FixedChunking(
                chunk_size=config.chunk_size,
                chunk_overlap=config.chunk_overlap,
            )
        else:
            raise ValueError(f"Unsupported chunk method: {config.chunk_method}")

        # Chunk the text (FixedChunking expects dict with "full_text" key)
        text_chunks = chunker.chunk({"full_text": text})

        # Parse chunks into expected format
        chunks = self._parse_chunks(text_chunks, chunk_id_prefix)
        logger.debug(f"Created {len(chunks)} chunks for processing")

        return chunks

    def _parse_chunks(
        self, chunks: List[str], chunk_id_prefix: str
    ) -> List[Dict[str, str]]:
        """
        Parse raw text chunks into structured format.

        Args:
            chunks: List of raw text chunks
            chunk_id_prefix: Prefix for chunk IDs

        Returns:
            List[Dict[str, str]]: Structured chunks with IDs
        """
        return [
            {"chunk": chunk, "chunk_id": f"{chunk_id_prefix}_{i}"}
            for i, chunk in enumerate(chunks)
        ]

    def _process_chunks_parallel(
        self,
        chunks: List[Dict[str, str]],
        kg_generator: KGGenerator,
        config: PipelineConfig,
    ) -> List[Dict[str, Any]]:
        """
        Process chunks in parallel to extract triplets.

        Args:
            chunks: List of text chunks to process
            kg_generator: Function to generate triplets from text
            config: Pipeline configuration

        Returns:
            List[Dict[str, Any]]: List of triplets from all chunks
        """

        # Helper function to process a single chunk
        def process_chunk(chunk_data):
            i, chunk = chunk_data
            logger.debug(f"Processing chunk {i + 1}/{len(chunks)}")

            # Distillation step
            if not config.skip_distillation:
                chunk["distilled"] = self.distill_text(
                    chunk["chunk"], model_name=config.distillation_model
                )
            else:
                chunk["distilled"] = chunk["chunk"]

            # Knowledge graph generation
            local_triplets = kg_generator(chunk["distilled"])
            local_triplets = find_triplets_in_text(
                chunk["chunk"], local_triplets, config.context_model
            )

            # Add chunk_id to each triplet
            for triplet in local_triplets:
                triplet["chunk_id"] = chunk["chunk_id"]

            logger.debug(
                f"Completed chunk {i + 1}/{len(chunks)} with {len(local_triplets)} triplets"
            )
            return local_triplets

        # Process chunks in parallel
        all_triplets = []
        max_workers = min(config.max_workers, len(chunks))

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all chunk processing tasks
            future_to_chunk = {
                executor.submit(process_chunk, (i, chunk)): i
                for i, chunk in enumerate(chunks)
            }

            # Collect results as they complete
            for future in as_completed(future_to_chunk):
                chunk_idx = future_to_chunk[future]
                try:
                    local_triplets = future.result()
                    all_triplets.extend(local_triplets)
                    logger.debug(
                        f"Added {len(local_triplets)} triplets to all_triplets (now: {len(all_triplets)})"
                    )
                except Exception as e:
                    logger.error(f"Error processing chunk {chunk_idx + 1}: {e}")
                    # Continue processing other chunks
                    continue

        return all_triplets
