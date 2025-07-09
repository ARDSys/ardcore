import sys
import time
from pathlib import Path

from loguru import logger
from openai import OpenAI

from ardcore.data.dataset import Dataset
from ardcore.data.research_paper import ResearchPaper

# No need to import extractors - TripletsGenerationPipeline uses ExtractorFactory automatically
from ardcore.knowledge_graph.knowledge_graph import KnowledgeGraph
from ardcore.knowledge_graph.node_merger.embedding_based import EmbeddingBasedNodeMerger

# --- Configuration ---
# Adjust these paths and configurations as needed for your setup

# 1. Paths
BASE_DIR = Path(__file__).parent
INITIAL_DATA_DIR = BASE_DIR / "data_initial"
INCREMENTAL_DATA_DIR = BASE_DIR / "data_incremental"
OUTPUT_DIR = BASE_DIR / "output_script"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 2. Triplet Extraction Configuration
# Choose one extraction_type: "refine", "review", "swarm"
# Ensure the corresponding generator is imported above.
TRIPLET_EXTRACTION_CONFIG = {
    "extraction_type": "refine",  # Example: using refine
    "extractor_model_name": "gpt-4o-mini",  # Directly specify model names
    "refiner_model_name": "gpt-4o-mini",  # Consistent with cli.py get_llm default
    "max_iterations": 3,  # Example value, adjust as needed
    "refiner_model_sleep": 0,  # Example value, adjust as needed
    # Add other parameters relevant to "refine" or other extractors
    # e.g., for refine: "schema": "..."
    # Refer to the respective extractor's documentation or cli.py for options
}

# 3. Node Merger Configuration
NODE_MERGER_CONFIG = {
    "embedding_model_name": "all-MiniLM-L6-v2",
    "similarity_threshold": 0.85,
}

# 4. Logging Setup
logger.remove()
logger.add(sys.stderr, level="INFO")
logger.add(OUTPUT_DIR / "full_graph_example.log", level="DEBUG")

# 5. OpenAI Client (if LLM-based extraction is used)
# Ensure your OPENAI_API_KEY environment variable is set
client = None
try:
    client = (
        OpenAI()
    )  # Initialize client, extractors might use it via utils.generate_response
    logger.info("OpenAI client initialized.")
except Exception as e:
    logger.error(
        f"Failed to initialize OpenAI client: {e}. Ensure OPENAI_API_KEY is set."
    )
    # Some extractors might fail if the client is not available.


# --- Helper Functions ---
def log_graph_state(kg: KnowledgeGraph, stage_name: str):
    """Logs the current state of the knowledge graph."""
    logger.info(f"--- Graph State: {stage_name} ---")
    logger.info(f"Nodes: {kg.number_of_nodes()}")
    logger.info(f"Edges: {kg.number_of_edges()}")
    logger.info(f"Edge Types: {len(kg.get_edges())}")
    logger.info("------------------------------------")


# No longer needed - TripletsGenerationPipeline handles extractor selection automatically


# --- Main Script ---


def main():
    logger.info("🚀 Starting Knowledge Graph Creation and Expansion Script 🚀")

    # --- 1. Setup ---
    logger.info("🔧 Initializing configurations and services...")
    # No need for manual generator setup - TripletsGenerationPipeline handles this automatically

    # Calculate kg_version from triplet extraction config
    # config_str = json.dumps(TRIPLET_EXTRACTION_CONFIG, sort_keys=True)
    # KG_VERSION = f"{TRIPLET_EXTRACTION_CONFIG['extraction_type']}_{hashlib.sha256(config_str.encode()).hexdigest()[:8]}"
    KG_VERSION = "baseline_1"  # Use pre-existing baseline_1 triplets
    logger.info(f"Using KG Version for items: {KG_VERSION}")

    node_merger = EmbeddingBasedNodeMerger(
        similarity_threshold=NODE_MERGER_CONFIG["similarity_threshold"],
    )
    logger.info(f"Node Merger Initialized: {NODE_MERGER_CONFIG}")

    # --- 2. Initial Knowledge Graph Creation ---
    logger.info("🌟 Section: Initial Knowledge Graph Creation 🌟")

    # 2.1. Load Initial Dataset
    logger.info(f"📂 Loading initial dataset from: {INITIAL_DATA_DIR}")
    from ardcore.storage.file import StorageManager

    initial_dataset = Dataset.from_local(
        INITIAL_DATA_DIR,
        daset_item_type=ResearchPaper,
        storage_manager=StorageManager("local", INITIAL_DATA_DIR),
        overwrite=True,
        kg_version=KG_VERSION,
    )
    logger.info(f"Found {len(initial_dataset.items)} items in the initial dataset.")

    # 2.2. Extract Triplets from Initial Dataset
    all_initial_triplets_objects = []
    logger.info("🔎 Extracting triplets from initial dataset...")
    start_time_initial_extraction = time.time()
    for i, item in enumerate(initial_dataset.items):
        logger.info(
            f"Processing initial item {i + 1}/{len(initial_dataset.items)}: {item.id}"
        )
        try:
            # item.generate_kg returns a Triplets object
            triplets_obj = item.generate_kg(
                kg_version=KG_VERSION,
                config=TRIPLET_EXTRACTION_CONFIG,  # Full config for the generator
                overwrite=False,  # Use pre-existing triplets if available
            )
            if triplets_obj and triplets_obj.triplets:
                all_initial_triplets_objects.append(triplets_obj)
                logger.info(
                    f"Extracted {len(triplets_obj.triplets)} triplets for item {item.id}"
                )
            else:
                logger.warning(f"No triplets extracted for item {item.id}")
        except Exception as e:
            logger.error(
                f"Error extracting triplets for item {item.id}: {e}", exc_info=True
            )

    elapsed_initial_extraction = time.time() - start_time_initial_extraction
    logger.info(
        f"⏱️ Initial triplet extraction completed in {elapsed_initial_extraction:.2f} seconds."
    )

    # 2.3. Build Initial Knowledge Graph
    logger.info("🏗️ Building initial knowledge graph...")
    kg = KnowledgeGraph(
        scientific_domain="example_scientific_domain"
    )  # Default backend is NetworkX

    # Add triplets from all Triplets objects
    for triplets_obj in all_initial_triplets_objects:
        kg.add_triplets(triplets_obj)  # add_triplets can take a Triplets object

    log_graph_state(kg, "Initial Graph - Before Merging")

    # 2.4. Perform Node Merging
    logger.info("🔗 Performing node merging on initial graph...")
    start_time_initial_merge = time.time()
    kg.merge_similar_nodes(node_merger)
    elapsed_initial_merge = time.time() - start_time_initial_merge
    logger.info(
        f"⏱️ Initial node merging completed in {elapsed_initial_merge:.2f} seconds."
    )
    log_graph_state(kg, "Initial Graph - After Merging")

    # 2.5. Save Initial Graph
    initial_graph_path = OUTPUT_DIR / "initial_graph.json"
    logger.info(f"💾 Saving initial graph to: {initial_graph_path}")
    kg.save_to_file(str(initial_graph_path))

    # --- 3. Incremental Knowledge Graph Expansion ---
    logger.info("📈 Section: Incremental Knowledge Graph Expansion 📈")

    # 3.1. Load Incremental Dataset
    logger.info(f"📂 Loading incremental dataset from: {INCREMENTAL_DATA_DIR}")
    incremental_dataset = Dataset.from_local(
        INCREMENTAL_DATA_DIR,
        daset_item_type=ResearchPaper,
        storage_manager=StorageManager("local", INITIAL_DATA_DIR),
        overwrite=True,
        kg_version=KG_VERSION,
    )
    logger.info(f"Found {len(incremental_dataset.items)} items for incremental update.")

    # 3.2. Process Each Incremental Item
    if not incremental_dataset.items:
        logger.info(
            "No items found in the incremental dataset. Skipping incremental update."
        )
    else:
        for i, item in enumerate(incremental_dataset.items):
            item_id_short = item.id.split("/")[-1][:8]  # Get a short, unique ID part
            logger.info(
                f"Processing incremental item {i + 1}/{len(incremental_dataset.items)}: {item.id} (Short ID: {item_id_short})"
            )
            start_time_item_processing = time.time()

            try:
                # a. Extract Triplets for the New Item
                logger.info(f"🔎 Extracting triplets for incremental item {item.id}...")
                new_triplets_obj = item.generate_kg(
                    kg_version=KG_VERSION,  # Use the same version for consistency
                    config=TRIPLET_EXTRACTION_CONFIG,
                    overwrite=False,  # Use pre-existing triplets if available
                )

                if not new_triplets_obj or not new_triplets_obj.triplets:
                    logger.warning(
                        f"No new triplets extracted for incremental item {item.id}. Skipping addition and merging for this item."
                    )
                    continue

                logger.info(
                    f"Extracted {len(new_triplets_obj.triplets)} new triplets for item {item.id}"
                )

                # b. Add Triplets to Existing Graph
                logger.info(
                    f"➕ Adding {len(new_triplets_obj.triplets)} new triplets to the graph..."
                )
                kg.add_triplets(new_triplets_obj)
                log_graph_state(
                    kg,
                    f"After Adding Triplets from Item {item_id_short} - Before Merging",
                )

                # c. Perform Node Merging
                logger.info(f"🔗 Performing node merging after item {item_id_short}...")
                start_time_item_merge = time.time()
                kg.merge_similar_nodes(node_merger)  # Re-use the same merger
                elapsed_item_merge = time.time() - start_time_item_merge
                logger.info(
                    f"⏱️ Node merging for item {item_id_short} completed in {elapsed_item_merge:.2f} seconds."
                )
                log_graph_state(kg, f"After Merging - Item {item_id_short}")

                # d. Save Iteration Graph (Optional)
                iteration_graph_path = (
                    OUTPUT_DIR / f"graph_after_item_{item_id_short}.json"
                )
                logger.info(f"💾 Saving graph to: {iteration_graph_path}")
                kg.save_to_file(str(iteration_graph_path))

            except Exception as e:
                logger.error(
                    f"Error processing incremental item {item.id}: {e}", exc_info=True
                )

            elapsed_item_processing = time.time() - start_time_item_processing
            logger.info(
                f"⏱️ Total processing for item {item_id_short} completed in {elapsed_item_processing:.2f} seconds."
            )

    # --- 4. Final Graph ---
    logger.info("🏁 Section: Final Graph State 🏁")
    log_graph_state(kg, "Final Graph")

    final_graph_path = OUTPUT_DIR / "final_graph.json"
    logger.info(f"💾 Saving final graph to: {final_graph_path}")
    kg.save_to_file(str(final_graph_path))

    logger.info("✅ Script execution completed successfully! ✅")


if __name__ == "__main__":
    main()
