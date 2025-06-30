# Full Graph Example: Creation and Incremental Update

## Conceptual Overview

This document demonstrates a complete workflow for creating and maintaining knowledge graphs from research papers. The process transforms unstructured text into structured knowledge that can support AI-driven scientific hypothesis generation and discovery.

### What is a Knowledge Graph?

A knowledge graph is a structured representation of information where:
- **Nodes** represent entities (e.g., "microglia", "inflammatory response", "Alzheimer's disease")
- **Edges** represent relationships between entities (e.g., "activates", "causes", "is_involved_in")
- **Triplets** are the basic building blocks: (subject, predicate, object) like ("microglia", "undergoes", "activation")

### Core Components

#### 1. Data Sources
- **Research Papers**: The primary input, stored as `ResearchPaper` objects
- **Text Processing**: Papers are chunked into manageable sections for analysis
- **Metadata**: Each paper includes structured information (DOI, authors, title, etc.)

#### 2. Triplet Extraction
The system converts natural language text into structured triplets using three main approaches:

- **Refine Method**: Uses an LLM to extract initial triplets, then iteratively refines them for consistency and accuracy
- **Review Method**: Extracts triplets and uses a reviewer LLM to evaluate quality across multiple dimensions (relevance, diversity, factualness, granularity, completeness)
- **Swarm Method**: Employs multiple LLM agents working collaboratively to extract and validate triplets

Each method is configurable and can use different LLM models for extraction and refinement phases.

#### 3. Knowledge Graph Construction
- **Graph Structure**: Triplets are assembled into a unified graph using NetworkX backend
- **Metadata Preservation**: Each node and edge retains information about its source (which paper, which chunk, confidence scores, etc.)
- **Incremental Addition**: New triplets can be added to existing graphs without rebuilding from scratch

#### 4. Node Merging & Deduplication
The system identifies and merges semantically similar nodes to create a coherent knowledge base:

- **Embedding-Based Merging**: Uses sentence transformers to compute semantic similarity between node names
- **Similarity Threshold**: Configurable threshold (e.g., 0.85) determines when nodes should be merged
- **Metadata Aggregation**: When nodes merge, their metadata and relationships are combined
- **Source Tracking**: The system maintains provenance information showing which papers contributed to each merged entity

#### 5. Versioning & Persistence
- **KG Versions**: Each extraction configuration creates a versioned set of triplets (e.g., "baseline_1")
- **Storage**: Graphs can be saved/loaded in JSON format for persistence and sharing
- **Incremental Updates**: New papers can be added without reprocessing existing content

### Workflow Stages

#### Stage 1: Initial Knowledge Graph Creation
1. **Setup**: Configure extraction methods, node merging parameters, and LLM models
2. **Data Loading**: Load initial dataset of research papers from local directory
3. **Triplet Extraction**: Process each paper to extract structured triplets using chosen method
4. **Graph Assembly**: Combine all triplets into a unified knowledge graph
5. **Node Merging**: Identify and merge semantically similar entities
6. **Persistence**: Save the initial graph state

#### Stage 2: Incremental Expansion
1. **New Data**: Load additional research papers for integration
2. **Per-Paper Processing**: For each new paper:
   - Extract triplets using the same configuration as initial creation
   - Add new triplets to existing graph
   - Perform node merging to integrate new entities with existing ones
   - Save intermediate graph state
3. **Final State**: Result is an expanded knowledge graph incorporating all available research

### Key Benefits

#### Scalability
- **Incremental Processing**: Add new papers without reprocessing existing content
- **Configurable Extraction**: Tune extraction methods for quality vs. speed trade-offs
- **Distributed Storage**: Supports both local and cloud storage backends

#### Quality Control
- **Multi-iteration Refinement**: Iterative improvement of triplet quality
- **Semantic Deduplication**: Automatic merging of equivalent entities across papers
- **Source Traceability**: Full provenance tracking for every piece of knowledge

#### Flexibility
- **Multiple Extraction Methods**: Choose the approach that best fits your domain and requirements
- **Configurable Parameters**: Tune similarity thresholds, chunk sizes, and model choices
- **Format Agnostic**: Works with various input formats and can output to multiple graph formats

### Use Cases

This knowledge graph infrastructure supports various downstream applications:
- **Scientific Hypothesis Generation**: AI agents can explore connections between entities to propose novel research directions
- **Literature Review**: Automated identification of key concepts and relationships across large paper collections
- **Knowledge Discovery**: Finding unexpected connections between disparate research areas
- **Research Gap Analysis**: Identifying under-explored relationships or entities in the literature

## 1. Setup

*   **Define Paths**:
    *   `INITIAL_DATA_DIR`: Path to `examples/full_graph_example/data_initial`
    *   `INCREMENTAL_DATA_DIR`: Path to `examples/full_graph_example/data_incremental`
    *   `OUTPUT_DIR`: Path for saving intermediate and final graph states (e.g., `examples/full_graph_example/output_script`)
*   **Define Configurations**:
    *   `TRIPLET_EXTRACTION_CONFIG`: Configuration for triplet extraction. This will be a dictionary, potentially including:
        *   `extraction_type`: (e.g., "refine", "review", "swarm")
        *   LLM model details (if applicable)
        *   Other parameters specific to the chosen extraction type.
    *   `NODE_MERGER_CONFIG`: Configuration for node merging:
        *   `embedding_model_name`: (e.g., "all-MiniLM-L6-v2")
        *   `similarity_threshold`: (e.g., 0.85)
*   **Initialize Services**:
    *   Set up logging (e.g., using `loguru`).
    *   Initialize an OpenAI client if LLM-based triplet extraction is used.
    *   Define a `get_llm` helper function if needed by extractors.

## 2. Initial Knowledge Graph Creation

### 2.1. Load Initial Dataset
    *   Use `Dataset.from_local(INITIAL_DATA_DIR, ResearchPaper)` to load items.
    *   Log the number of items found.

### 2.2. Extract Triplets from Initial Dataset
    *   Define a `triplets_generator` function based on `TRIPLET_EXTRACTION_CONFIG`. This function will wrap one of the `extract_..._generator` functions from `ard.data.triplets_extractor`.
    *   Calculate a `kg_version` string based on a hash of the `TRIPLET_EXTRACTION_CONFIG` (similar to `cli.py`) to ensure consistent versioning if data is stored by items.
    *   Iterate through each item in the initial dataset:
        *   Log processing of the current item.
        *   Call `item.generate_kg(kg_version=kg_version, kg_generator=triplets_generator, config=TRIPLET_EXTRACTION_CONFIG, overwrite=True)` to get `Triplets`.
        *   Store the returned `Triplets` objects.
    *   Collect all generated `Triplets` objects into a single list.

### 2.3. Build Initial Knowledge Graph
    *   Create an empty `KnowledgeGraph` instance: `kg = KnowledgeGraph()`.
    *   Add the collected triplets to the graph: `kg.add_triplets(all_initial_triplets)`.
    *   Log the state of the graph (nodes, edges) before merging.

### 2.4. Perform Node Merging
    *   Initialize an `EmbeddingBasedNodeMerger` using `NODE_MERGER_CONFIG` (e.g., `EmbeddingBasedNodeMerger(embedding_model_name=..., similarity_threshold=...)`).
    *   Call `kg.merge_similar_nodes(merger)` on the graph.
    *   Log the state of the graph after merging.

### 2.5. Save Initial Graph (Optional)
    *   Create the `OUTPUT_DIR` if it doesn't exist.
    *   Save the graph to a file: `kg.save_to_file(Path(OUTPUT_DIR) / "initial_graph.json")`.

## 3. Incremental Knowledge Graph Expansion

### 3.1. Load Incremental Dataset
    *   Use `Dataset.from_local(INCREMENTAL_DATA_DIR, ResearchPaper)` to load items.
    *   Log the number of items found for incremental update.

### 3.2. Process Each Incremental Item
    *   Use the same `triplets_generator`, `TRIPLET_EXTRACTION_CONFIG`, and `kg_version` from step 2.2.
    *   Use the same `EmbeddingBasedNodeMerger` instance (`merger`) from step 2.4.
    *   Iterate through each item in the incremental dataset:
        *   Log processing of the current incremental item.
        *   **a. Extract Triplets for the New Item**:
            *   Generate `Triplets` for the current item: `new_triplets = item.generate_kg(...)`.
        *   **b. Add Triplets to Existing Graph**:
            *   Call `kg.add_triplets(new_triplets)`.
            *   Log graph state after adding new triplets but before merging.
        *   **c. Perform Node Merging**:
            *   Call `kg.merge_similar_nodes(merger)` again on the updated graph.
            *   Log graph state after merging.
        *   **d. Save Iteration Graph (Optional)**:
            *   Save the graph: `kg.save_to_file(Path(OUTPUT_DIR) / f"graph_after_item_{item.id[:8]}.json")`.

## 4. Final Graph

*   The `kg` object now contains the combined and merged knowledge.
*   Log the final state of the graph.
*   Save the final graph: `kg.save_to_file(Path(OUTPUT_DIR) / "final_graph.json")`.

## Notes on Implementation:

*   Ensure all necessary imports are at the top of `example.py`.
*   Structure the script with clear functions or sections for each major step (Initial Creation, Incremental Update).
*   Use logging extensively to track progress and intermediate states.
*   Make sure paths are constructed correctly using `pathlib.Path`.
