import time
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from loguru import logger
from openai import OpenAI

from ardcore.knowledge_graph.knowledge_graph import KnowledgeGraph
from ardcore.storage.file import StorageManager
from ardcore.subgraph import Subgraph
from ardcore.subgraph.subgraph_generator import (
    RandomizedEmbeddingPathGenerator,
)
from ardcore.subgraph.subgraph_generator.embedding import EmbeddingPathGenerator
from ardcore.subgraph.subgraph_generator.llm_walk import LLMWalkGenerator
from ardcore.subgraph.subgraph_generator.random_walk import (
    SingleNodeRandomWalkGenerator,
)
from ardcore.subgraph.subgraph_generator.shortest_path import ShortestPathGenerator
from ardcore.utils.embedder import Embedder

# --- Configuration ---
BASE_DIR = Path(__file__).parent
INPUT_KG_PATH = BASE_DIR.parent / "full_graph_example/output_script/final_graph.json"
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def get_llm(model: str = "gpt-4o-mini"):
    client = OpenAI()

    if model == "small":
        model = "gpt-4o-mini"
    elif model == "large":
        model = "gpt-4o"
    elif model == "reasoning":
        model = "o3-mini"

    def llm(prompt: str):
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content

    return llm


def log_timing(operation_name, start_time):
    """Log the time taken for an operation in a consistent format."""
    elapsed = time.time() - start_time
    logger.info(f"⏱️ {operation_name} completed in {elapsed:.2f} seconds")


def log_section(section_name):
    """Create a visual separator for a new section in logs."""
    logger.info(f"\n{'=' * 50}")
    logger.info(f"📌 {section_name}")
    logger.info(f"{'=' * 50}")


def save_subgraph(
    subgraph: Subgraph,
    output_name: str,
    storage_path: Optional[Path] = None,
):
    """Save a subgraph to local storage."""
    storage_manager = StorageManager(
        storage_type="local",
        storage_path=storage_path,
        storage_name="save_subgraph",
    )
    storage = storage_manager.get_backend(name="save_subgraph")

    subgraph.save_to_file(
        f"{output_name}.subgraph.json", storage=storage, item_id=output_name
    )
    logger.info(f"✅ Saved subgraph to {output_name}.subgraph.json at {storage_path}")


def create_subgraph_with_method(
    kg: KnowledgeGraph,
    method: str,
    max_steps: int,
    max_nodes: int,
    neighbor_probability: float,
    embedder: Embedder | None = None,
    llm: str = "small",
):
    """Create a subgraph using the specified method."""
    if method == "random_walk":
        start_node = kg.get_random_node()
        logger.info(f"   Starting from node: {start_node}")
        subgraph = Subgraph.from_one_node(
            kg,
            start_node,
            method=SingleNodeRandomWalkGenerator(max_steps=max_steps),
            max_nodes=max_nodes,
            neighbor_probability=neighbor_probability,
        )
    elif method == "llm_walk":
        start_node = kg.get_random_node()
        logger.info(f"   Starting from node: {start_node}")
        subgraph = Subgraph.from_one_node(
            kg,
            start_node,
            method=LLMWalkGenerator(max_steps=max_steps, llm=get_llm(llm)),
            max_nodes=max_nodes,
            neighbor_probability=neighbor_probability,
        )
    elif method == "embedding_path":
        start_node = kg.get_random_node()
        end_node = kg.get_random_node()
        logger.info(f"   Path from: {start_node} to {end_node}")
        subgraph = Subgraph.from_two_nodes(
            kg,
            start_node,
            end_node,
            method=EmbeddingPathGenerator(embedder=embedder),
            max_nodes=max_nodes,
            neighbor_probability=neighbor_probability,
        )
    elif method == "randomized_embedding_path":
        start_node = kg.get_random_node()
        end_node = kg.get_random_node()
        logger.info(f"   Path from: {start_node} to {end_node}")
        subgraph = Subgraph.from_two_nodes(
            kg,
            start_node,
            end_node,
            method=RandomizedEmbeddingPathGenerator(embedder=embedder),
            max_nodes=max_nodes,
            neighbor_probability=neighbor_probability,
        )
    elif method == "shortest_path":
        start_node = kg.get_random_node()
        end_node = kg.get_random_node()
        logger.info(f"   Path from: {start_node} to {end_node}")
        subgraph = Subgraph.from_two_nodes(
            kg,
            start_node,
            end_node,
            method=ShortestPathGenerator(),
            max_nodes=max_nodes,
            neighbor_probability=neighbor_probability,
        )
    else:
        raise ValueError(f"Invalid method: {method}")

    return subgraph


def generate_subgraph(
    scientific_domain: str,
    graph_path=None,
    embedder_path=None,
    max_nodes=10,
    max_steps=5,
    method="llm_walk",
    min_score=3,
    neighbor_probability=0.2,
    llm="large",
    max_attempts=3,
):
    """Generate a single subgraph with context from a knowledge graph."""
    log_section("SUBGRAPH GENERATION")
    logger.info("📊 Configuration:")
    logger.info(f"   Method: {method}")
    logger.info(f"   Max nodes: {max_nodes}")
    logger.info(f"   Max steps: {max_steps}")
    logger.info(f"   Minimum score: {min_score}")
    logger.info(f"   Neighbor probability: {neighbor_probability}")
    logger.info(f"   LLM: {llm}")
    logger.info(f"   Max attempts: {max_attempts}")

    # Initialize embedder if provided
    embedder = Embedder()
    if embedder_path:
        logger.info(f"🔄 Loading embedder from {embedder_path}")
        embedder.load_from_file(embedder_path)
    else:
        logger.info("ℹ️ No embedder provided, will compute embeddings on the fly")

    # Load knowledge graph
    logger.info("🔄 Loading knowledge graph...")
    start_time = time.time()

    logger.info(f"📊 Loading knowledge graph from file: {graph_path}")
    kg = KnowledgeGraph.load_from_file(graph_path, scientific_domain=scientific_domain)

    log_timing("Knowledge graph loading", start_time)
    logger.info("✅ Loaded graph successfully")

    # Generate a single subgraph
    log_section("GENERATING SUBGRAPH")
    subgraph = None
    attempt = 0

    while attempt < max_attempts:
        attempt += 1
        try:
            logger.info(
                f"🔄 Attempt {attempt}: Generating subgraph with method '{method}'"
            )
            start_time = time.time()

            # Create the subgraph using the specified method
            subgraph = create_subgraph_with_method(
                kg=kg,
                method=method,
                max_steps=max_steps,
                max_nodes=max_nodes,
                neighbor_probability=neighbor_probability,
                embedder=embedder,
                llm=llm,
            )

            log_timing("Subgraph generation", start_time)

            logger.info("🔄 Scoring subgraph path...")
            start_time = time.time()
            subgraph.score_path(get_llm(llm))
            log_timing("Path scoring", start_time)

            score_info = f"Score: {subgraph._path_score:.2f}"
            if subgraph._path_score < min_score:
                logger.warning(
                    f"⚠️ {score_info} - Below minimum threshold of {min_score}"
                )
                subgraph = None
            else:
                logger.info(f"✅ {score_info} - Above minimum threshold")
                logger.info("🔄 Contextualizing subgraph...")
                start_time = time.time()
                subgraph.contextualize(llm=get_llm(llm))
                log_timing("Contextualization", start_time)
                break

        except Exception as e:
            logger.error(f"❌ Error in attempt {attempt}: {e}")
            subgraph = None

        if attempt >= max_attempts and subgraph is None:
            logger.warning(f"⚠️ Failed after {attempt} attempts")
        elif subgraph is None:
            logger.info("🔄 Retrying...")

    # Save the subgraph if generated successfully
    if subgraph:
        logger.info("💾 Saving subgraph with context")
        output_name = subgraph.subgraph_id
        save_subgraph(subgraph, output_name, storage_path=OUTPUT_DIR)

        log_section("SUBGRAPH SUMMARY")
        logger.info(f"Start node: {subgraph.start_node}")
        logger.info(f"End node: {subgraph.end_node}")
        logger.info(f"Number of nodes: {subgraph.number_of_nodes()}")
        logger.info(f"Number of edges: {subgraph.number_of_edges()}")
        logger.info(f"Path score: {subgraph._path_score:.2f}")

        return subgraph
    else:
        logger.error("❌ No valid subgraph was generated")
        return None


def main():
    """Main function to run the example."""
    load_dotenv()
    generate_subgraph(
        scientific_domain="example_scientific_domain",
        graph_path=INPUT_KG_PATH,
        max_nodes=20,
        max_steps=5,
        method="llm_walk",
        min_score=3,
    )


if __name__ == "__main__":
    main()
