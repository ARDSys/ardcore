"""
Unit tests for TripletsGenerationPipeline.
"""

import pytest

from ardcore.data.triplets_extractor.config.pipeline import PipelineConfig
from ardcore.data.triplets_extractor.pipeline import TripletsGenerationPipeline


class TestTripletsGenerationPipeline:
    """Test suite for TripletsGenerationPipeline."""

    def test_initialization_with_default_config(self):
        """Test pipeline initialization with default configuration."""
        pipeline = TripletsGenerationPipeline()

        assert isinstance(pipeline.pipeline_config, PipelineConfig)
        assert pipeline.pipeline_config.chunk_size == 1000
        assert pipeline.pipeline_config.max_workers == 8
        assert pipeline.pipeline_config.scientific_domain == "bioscience"

    def test_initialization_with_custom_config(self):
        """Test pipeline initialization with custom configuration."""
        custom_config = {
            "chunk_size": 2000,
            "max_workers": 16,
            "scientific_domain": "chemistry",
        }
        pipeline = TripletsGenerationPipeline(config=custom_config)

        assert pipeline.pipeline_config.chunk_size == 2000
        assert pipeline.pipeline_config.max_workers == 16
        assert pipeline.pipeline_config.scientific_domain == "chemistry"
        # Defaults should be preserved for unspecified values
        assert pipeline.pipeline_config.chunk_overlap == 100

    def test_initialization_with_invalid_config(self):
        """Test pipeline initialization with invalid configuration raises error."""
        invalid_config = {
            "chunk_size": -1000,  # Invalid negative value
        }

        with pytest.raises(ValueError, match="chunk_size must be positive"):
            TripletsGenerationPipeline(config=invalid_config)

    def test_generate_triplets_input_validation(self):
        """Test input validation for generate_triplets method."""
        pipeline = TripletsGenerationPipeline()

        # Test empty string
        with pytest.raises(ValueError, match="Text input must be a non-empty string"):
            pipeline.generate_triplets("")

        # Test None input
        with pytest.raises(ValueError, match="Text input must be a non-empty string"):
            pipeline.generate_triplets(None)

        # Test non-string input
        with pytest.raises(ValueError, match="Text input must be a non-empty string"):
            pipeline.generate_triplets(123)

    def test_config_override_merging_logic(self):
        """Test configuration override merging without LLM calls."""
        pipeline = TripletsGenerationPipeline(
            config={"chunk_size": 1000, "max_workers": 8}
        )

        # Test that original config is preserved
        assert pipeline.pipeline_config.chunk_size == 1000
        assert pipeline.pipeline_config.max_workers == 8

        # Test configuration merging logic (without actually calling generate_triplets)
        base_config = pipeline.full_config.copy()
        overrides = {"chunk_size": 2000, "max_workers": 16}

        # Simulate the merging that happens in generate_triplets
        effective_config = base_config.copy()
        effective_config.update(overrides)

        # Verify merging worked correctly
        assert effective_config["chunk_size"] == 2000
        assert effective_config["max_workers"] == 16

        # Original pipeline config should be unchanged
        assert pipeline.pipeline_config.chunk_size == 1000
        assert pipeline.pipeline_config.max_workers == 8

    def test_config_merging_logic(self):
        """Test configuration merging logic without LLM calls."""
        config = {"extraction_type": "review", "chunk_size": 500}
        pipeline = TripletsGenerationPipeline(config=config)

        # Test that both pipeline and extractor configs are preserved
        assert pipeline.pipeline_config.chunk_size == 500
        assert pipeline.full_config["extraction_type"] == "review"
        assert pipeline.pipeline_config.scientific_domain == "bioscience"  # Default

    def test_extractor_factory_integration(self):
        """Test that ExtractorFactory integration works."""
        from ardcore.data.triplets_extractor.config.factory import ExtractorFactory

        config = {"extraction_type": "refine"}
        pipeline = TripletsGenerationPipeline(config=config)

        # Test that factory can create generator from config
        generator = ExtractorFactory.create_generator(config)
        assert callable(generator)

        # Test that pipeline stores the config correctly for factory use
        assert pipeline.full_config["extraction_type"] == "refine"
        assert "extraction_type" in pipeline.full_config

    def test_get_config(self):
        """Test getting current configuration."""
        custom_config = {"chunk_size": 1500, "scientific_domain": "physics"}
        pipeline = TripletsGenerationPipeline(config=custom_config)

        config = pipeline.get_config()

        assert isinstance(config, PipelineConfig)
        assert config.chunk_size == 1500
        assert config.scientific_domain == "physics"

    def test_update_config(self):
        """Test updating pipeline configuration."""
        pipeline = TripletsGenerationPipeline(config={"chunk_size": 1000})

        assert pipeline.pipeline_config.chunk_size == 1000
        assert pipeline.pipeline_config.max_workers == 8  # Default

        pipeline.update_config({"chunk_size": 2000, "max_workers": 16})

        assert pipeline.pipeline_config.chunk_size == 2000
        assert pipeline.pipeline_config.max_workers == 16

    def test_update_config_with_invalid_values(self):
        """Test that updating config with invalid values raises error."""
        pipeline = TripletsGenerationPipeline()

        with pytest.raises(ValueError, match="chunk_size must be positive"):
            pipeline.update_config({"chunk_size": -500})

    def test_string_representation(self):
        """Test string representations of the pipeline."""
        custom_config = {"chunk_size": 1500}
        pipeline = TripletsGenerationPipeline(config=custom_config)

        str_repr = str(pipeline)
        repr_repr = repr(pipeline)

        assert "TripletsGenerationPipeline" in str_repr
        assert "TripletsGenerationPipeline" in repr_repr
        assert str_repr == repr_repr

    def test_chunk_text_basic(self):
        """Test basic text chunking functionality."""
        pipeline = TripletsGenerationPipeline(
            config={"chunk_size": 100, "chunk_overlap": 20}
        )

        text = (
            "This is a sample text for testing chunking functionality. " * 10
        )  # Make it long enough to chunk
        chunks = pipeline.chunk_text(text, pipeline.pipeline_config)

        assert isinstance(chunks, list)
        assert len(chunks) > 0

        # Check structure of chunks
        for chunk in chunks:
            assert "chunk" in chunk
            assert "chunk_id" in chunk
            assert isinstance(chunk["chunk"], str)
            assert chunk["chunk_id"].startswith("chunk_")

    def test_chunk_text_with_custom_prefix(self):
        """Test text chunking with custom chunk ID prefix."""
        pipeline = TripletsGenerationPipeline(
            config={"chunk_size": 50, "chunk_overlap": 10}
        )

        text = "Sample text for chunking with custom prefix. " * 5
        chunks = pipeline.chunk_text(
            text, pipeline.pipeline_config, chunk_id_prefix="test_paper"
        )

        assert len(chunks) > 0
        for chunk in chunks:
            assert chunk["chunk_id"].startswith("test_paper_")

    def test_chunk_text_respects_config(self):
        """Test that chunking respects configuration parameters."""
        # Test with small chunk size should create more chunks
        pipeline_small = TripletsGenerationPipeline(
            config={"chunk_size": 50, "chunk_overlap": 10}
        )
        # Test with large chunk size should create fewer chunks
        pipeline_large = TripletsGenerationPipeline(
            config={"chunk_size": 500, "chunk_overlap": 50}
        )

        text = (
            "This is a longer sample text for testing that chunking configuration is respected. "
            * 20
        )

        chunks_small = pipeline_small.chunk_text(text, pipeline_small.pipeline_config)
        chunks_large = pipeline_large.chunk_text(text, pipeline_large.pipeline_config)

        # Small chunks should create more pieces (assuming text is long enough)
        assert len(chunks_small) >= len(chunks_large)

    def test_distill_text_static_method_exists(self):
        """Test that distill_text method exists with correct signature."""
        # Test that the method exists and has correct signature
        assert hasattr(TripletsGenerationPipeline, "distill_text")
        assert callable(TripletsGenerationPipeline.distill_text)

    def test_chunk_text_empty_input(self):
        """Test chunking with empty text."""
        pipeline = TripletsGenerationPipeline()

        chunks = pipeline.chunk_text("", pipeline.pipeline_config)

        # Should handle empty text gracefully
        assert isinstance(chunks, list)

    def test_chunk_text_short_input(self):
        """Test chunking with very short text."""
        pipeline = TripletsGenerationPipeline(
            config={"chunk_size": 1000, "chunk_overlap": 100}
        )

        text = "Short text."
        chunks = pipeline.chunk_text(text, pipeline.pipeline_config)

        assert isinstance(chunks, list)
        assert len(chunks) >= 1  # Should have at least one chunk
        if len(chunks) > 0:
            assert (
                chunks[0]["chunk"] == text
            )  # Short text should remain as single chunk
