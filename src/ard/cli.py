import csv
import hashlib
import json
import time
from pathlib import Path, PurePosixPath

import click
import pandas as pd
from loguru import logger
from openai import OpenAI

from ard.data.dataset import Dataset
from ard.data.research_paper import ResearchPaper
from ard.data.triplets import Triplets

# Extractor imports no longer needed - ExtractorFactory handles this automatically
from ard.knowledge_graph import KnowledgeGraph
from ard.knowledge_graph.node_merger.embedding_based import EmbeddingBasedNodeMerger
from ard.storage.file import LocalStorageBackend, S3StorageBackend, StorageManager
from ard.subgraph.subgraph import Subgraph
from ard.subgraph.subgraph_generator import (
    RandomizedEmbeddingPathGenerator,
)
from ard.subgraph.subgraph_generator.embedding import EmbeddingPathGenerator
from ard.subgraph.subgraph_generator.llm_walk import LLMWalkGenerator
from ard.subgraph.subgraph_generator.random_walk import (
    SingleNodeRandomWalkGenerator,
)
from ard.subgraph.subgraph_generator.shortest_path import ShortestPathGenerator
from ard.utils.embedder import Embedder

client = OpenAI()


def get_llm(model: str = "gpt-4o-mini"):
    if model == "small":
        model = "gpt-4o-mini"
    elif model == "large":
        model = "gpt-4o"
    elif model == "reasoning":
        model = "o3-mini"

    def llm(prompt: str):
        response = client.responses.create(
            model=model,
            input=prompt,
        )
        return response.output_text

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


@click.group()
def cli():
    """ARD - Knowledge Graph and Subgraph Pipeline Tool."""
    pass


@cli.command("trc")
@click.option(
    "--data-path",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    help="Path to the data directory. If not provided, uses the default example data.",
)
@click.option(
    "--max-items",
    type=int,
    default=10,
    help="Maximum number of items to process (default: 10)",
)
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, dir_okay=True),
    default="output/triplets",
    help="Output directory for saving triplets",
)
@click.option(
    "--kg-version",
    type=str,
    default=None,
    help="KG version to use (default: None means all versions)",
)
@click.option(
    "--output-format",
    type=click.Choice(["markdown", "csv", "xlsx"]),
    default="markdown",
    help="Output format for saving triplets (default: markdown)",
)
def add_triplets_context(data_path, max_items, output_dir, kg_version, output_format):
    """Add context to triplets."""
    log_section("ADD TRIPLETS CONTEXT")
    logger.info(f"📂 Using provided data directory: {data_path}")

    # Determine data directory
    if data_path:
        data_dir = Path(data_path)
        logger.info(f"📂 Using provided data directory: {data_dir}")
    else:
        data_dir = Path(__file__).parent.parent.parent.parent / "data"
        logger.info(f"📂 Using default example data directory: {data_dir}")

    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"📁 Output directory: {output_path}")

    # Read a Dataset from the directory
    logger.info("🔄 Reading dataset from directory...")
    storage_manager = StorageManager()
    storage_manager.add_backend(storage_type="local", storage_path=data_dir)

    dataset = Dataset.from_local(
        path=data_dir,
        daset_item_type=ResearchPaper,
        storage_manager=storage_manager,
        kg_version=kg_version,
    )
    logger.info(f"Processing {max_items or len(dataset)} items")
    for item in dataset.items[:max_items]:
        if kg_version is None:
            versions = item.list_kg_versions()
        else:
            versions = [kg_version]

        for version in versions:
            logger.info(
                f"Extracting triplets for item {item.id} with version: {version}"
            )
            triplets = item.add_context_to_triplets(version)
            # Save triplets to markdown file
            output_file = output_path / f"{item.id[:8]}_{version}.md"
            logger.info(f"Saving triplets to {output_file}")
            save_triplets(triplets, item, output_file, output_format)


def save_triplets(
    triplets: Triplets,
    item: ResearchPaper,
    path: Path,
    output_format: str,
    storage_manager: StorageManager,
):
    """Save triplets to a file.

    Args:
        triplets: Triplets object containing the triplets, config, and metadata
        path: Path object for the output file
        output_format: Format to save the triplets in
        storage_manager: StorageManager object
    """
    if output_format == "markdown":
        save_triplets_to_markdown(
            triplets, item, path.with_suffix(".md"), storage_manager
        )
    elif output_format == "csv":
        save_triplets_to_csv(triplets, item, path.with_suffix(".csv"), storage_manager)
    elif output_format == "xlsx":
        save_triplets_to_xlsx(
            triplets, item, path.with_suffix(".xlsx"), storage_manager
        )


def save_triplets_to_xlsx(
    triplets: Triplets, item: ResearchPaper, path: Path, storage_manager: StorageManager
):
    """Save triplets to an xlsx file.

    Args:
        triplets: Triplets object containing the triplets, config, and metadata
        path: Path object for the output file
    """
    # Create dataframe for triplets
    data = []
    for triplet in triplets.triplets:
        data.append(
            {
                "node_1": triplet.node_1,
                "edge": triplet.edge,
                "node_2": triplet.node_2,
                "snippet": triplet.metadata.get("snippet", ""),
            }
        )

    triplets_df = pd.DataFrame(data)

    # Create a dataframe for metadata
    metadata_df = pd.DataFrame([triplets.item_metadata])

    # Create a dataframe for config
    config_df = pd.DataFrame([triplets.config])

    # Create Excel writer object
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        triplets_df.to_excel(writer, sheet_name="Triplets", index=False)
        metadata_df.to_excel(writer, sheet_name="Metadata", index=False)
        config_df.to_excel(writer, sheet_name="Configuration", index=False)

        # Add full text to a separate sheet if available
        processed_data = item.get_processed_data()
        full_text_df = pd.DataFrame([{"full_text": processed_data["full_text"]}])
        full_text_df.to_excel(writer, sheet_name="Full Text", index=False)


def save_triplets_to_markdown(
    triplets: Triplets, item: ResearchPaper, path: Path, storage_manager: StorageManager
):
    """Save triplets to a markdown file.

    Args:
        triplets: Triplets object containing the triplets, config, and metadata
        path: Path object for the output file
    """
    with open(path, "w", encoding="utf-8") as f:
        # Write header
        f.write("# Knowledge Graph Triplets\n\n")

        # Write config section
        f.write("## Configuration\n\n")
        f.write("```json\n")
        f.write(json.dumps(triplets.config, indent=2))
        f.write("\n```\n\n")

        # Write metadata section
        f.write("## Item Metadata\n\n")
        f.write("```json\n")
        f.write(json.dumps(triplets.item_metadata, indent=2))
        f.write("\n```\n\n")

        # Write triplets section
        f.write("## Triplets\n\n")
        for triplet in triplets.triplets:
            f.write(f"### {triplet.node_1} -[{triplet.edge}]-> {triplet.node_2}\n")
            f.write(f"**Snippet:** {triplet.metadata.get('snippet', '')}\n")

        # Write full article text if available
        processed_data = item.get_processed_data()
        f.write("\n## Full Article Text\n\n")
        f.write("```\n")
        f.write(processed_data["full_text"])
        f.write("\n```\n")


def save_triplets_to_csv(
    triplets: Triplets, item: ResearchPaper, path: Path, storage_manager: StorageManager
):
    """Save triplets to a csv file.

    Args:
        triplets: Triplets object containing the triplets, config, and metadata
        path: Path object for the output file
    """
    with open(path, "w", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["node_1", "edge", "node_2", "snippet"])
        for triplet in triplets.triplets:
            writer.writerow(
                [
                    triplet.node_1,
                    triplet.edge,
                    triplet.node_2,
                    triplet.metadata.get("snippet", ""),
                ]
            )


def save_subgraph_with_tweets(
    subgraph: Subgraph, tweets: dict[str, str] | None, path: Path
):
    """Save a subgraph and associated tweets to files."""
    subgraph.save_to_file(path.with_suffix(".subgraph.pkl"))
    subgraph.save_to_json(path.with_suffix(".subgraph.json"))
    with open(path.with_suffix(".tweets.md"), "w", encoding="utf-8") as f:
        f.write(f"# Subgraph\n\n{subgraph}\n\n")
        if tweets is not None:
            f.write("## Tweets\n\n")
            for tweet_type, tweet in tweets.items():
                f.write(f"# {tweet_type}\n{tweet}\n\n")
        else:
            f.write("## No tweets generated\n\n")
        f.write(f"## Path\n\n{subgraph.to_cypher_string()}\n\n")
        f.write(
            f"## Path Score\n\nScore: {subgraph._path_score}\nJustification: {subgraph._path_score_justification}\n\n"
        )
        f.write(f"## Context\n\n{subgraph.context}\n\n")


@cli.command("trx")
@click.option(
    "--storage-type",
    type=click.Choice(["local", "s3"]),
    default="local",
    help="Storage type to use (default: local)",
)
@click.option(
    "--data-path",
    type=str,
    help="Path to the data directory. If not provided, uses the default example data.",
)
@click.option(
    "--config-file",
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
    default="./triplets/config.json",
    help="Path to the config file",
)
@click.option(
    "--max-items",
    type=int,
    default=10,
    help="Maximum number of items to process (default: 10)",
)
@click.option(
    "--output-dir",
    type=str,
    default="output/triplets",
    help="Output directory for saving triplets",
)
@click.option(
    "--overwrite",
    "-o",
    is_flag=True,
    help="Overwrite existing triplets",
)
@click.option(
    "--output-format",
    type=click.Choice(["markdown", "csv", "xlsx"]),
    default="markdown",
    help="Output format for saving triplets (default: markdown)",
)
def extract_triplets(
    storage_type,
    data_path,
    config_file,
    max_items,
    output_dir,
    overwrite,
    output_format,
):
    """Create a knowledge graph from data."""
    log_section("TRIPLETS EXTRACTION")
    logger.info(f"📂 Using provided config file: {config_file}")
    with open(config_file, "r", encoding="utf-8") as f:
        config = json.load(f)

    extraction_type = config.get("extraction_type")
    if not extraction_type:
        raise ValueError("extraction_type is required in config file")

    # calculate the hash of the config
    config_hash = hashlib.sha256(json.dumps(config).encode()).hexdigest()[:8]
    kg_version = f"{extraction_type}_{config_hash}"

    # Determine data directory
    if data_path:
        data_dir = PurePosixPath(data_path)
        logger.info(f"📂 Using provided data directory: {data_dir}")
    else:
        data_dir = Path(__file__).parent.parent.parent.parent / "data"
        logger.info(f"📂 Using default example data directory: {data_dir}")

    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"📁 Output directory: {output_path}")

    # Create a Dataset from the directory
    logger.info("🔄 Creating dataset from ResearchPaper")
    storage_manager = StorageManager()

    if storage_type == "local":
        storage_manager.register_backend("local", LocalStorageBackend(data_dir))
        dataset = Dataset.from_local(
            data_dir, ResearchPaper, storage_manager, overwrite, kg_version
        )
    elif storage_type == "s3":
        storage_manager.register_backend("s3", S3StorageBackend(data_dir))
        dataset = Dataset.from_s3(
            data_dir, ResearchPaper, storage_manager, overwrite, kg_version
        )
    logger.info(
        f"Processing {max_items or len(dataset)} items with KG version: {kg_version}"
    )
    for item in dataset.items[:max_items]:
        logger.info(
            f"Extracting triplets for item {item.id} with version: {kg_version}"
        )
        triplets = item.generate_kg(
            kg_version=kg_version,
            config=config,
            overwrite=overwrite,
        )

        if storage_type == "local":
            # Save triplets to markdown file
            output_file = output_path / f"{item.id[:8]}_{kg_version}"
            logger.info(f"Saving triplets to {output_file}")
            save_triplets(triplets, item, output_file, output_format, storage_manager)


@cli.command("graph")
@click.option(
    "--data-path",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    help="Path to the data directory. If not provided, uses the default example data.",
)
@click.option(
    "--output",
    type=click.Path(),
    default="knowledge_graph.json",
    help="Path to save the knowledge graph (default: knowledge_graph.json)",
)
@click.option(
    "--max-items",
    type=int,
    default=10,
    help="Maximum number of items to process (default: 10)",
)
@click.option(
    "--similarity-threshold",
    "-st",
    type=float,
    default=0.85,
    help="Similarity threshold for merging nodes (default: 0.85)",
)
@click.option(
    "--overwrite",
    "-o",
    is_flag=True,
    help="Overwrite existing knowledge graph",
)
@click.option(
    "--kg-version",
    "-kgv",
    type=str,
    default="baseline_1",
    help="KG version to use (default: baseline_1)",
)
@click.option(
    "--embedder-path",
    "-ep",
    type=click.Path(file_okay=True, dir_okay=False),
    default=None,
    help="Path to the embedder file. If not provided, it computes the embeddings on the fly.",
)
@click.option(
    "--candidate-finder-strategy",
    "-cfs",
    type=click.Choice(["brute_force", "ann"]),
    default="brute_force",
    help="Candidate finder strategy to use (default: brute_force)",
)
@click.option(
    "--ann-k",
    "-ak",
    type=int,
    default=50,
)
def create_graph(
    data_path,
    output,
    max_items,
    similarity_threshold,
    overwrite,
    kg_version,
    embedder_path,
    candidate_finder_strategy,
    ann_k,
):
    """Create a knowledge graph from data."""
    log_section("KNOWLEDGE GRAPH CREATION")

    # Determine data directory
    if data_path:
        data_dir = Path(data_path)
        logger.info(f"📂 Using provided data directory: {data_dir}")
    else:
        data_dir = Path(__file__).parent.parent.parent.parent / "data"
        logger.info(f"📂 Using default example data directory: {data_dir}")

    storage_manager = StorageManager("local", data_dir)

    # Create a Dataset from the directory
    logger.info("🔄 Creating dataset from directory...")
    dataset = Dataset.from_local(
        data_dir,
        daset_item_type=ResearchPaper,
        storage_manager=storage_manager,
        overwrite=overwrite,
        kg_version=kg_version,
    )

    # Create a KnowledgeGraph from triplets
    log_section("BUILDING GRAPH")
    logger.info(f"🔄 Building knowledge graph (max_items={max_items})...")
    start_time = time.time()
    kg = KnowledgeGraph.from_dataset(dataset, max_items=max_items)
    log_timing("Knowledge graph creation", start_time)
    logger.debug("Knowledge Graph details:")
    logger.debug(kg)

    # Merge similar nodes using the EmbeddingBasedNodeMerger
    log_section("MERGING SIMILAR NODES")

    logger.info(f"🔄 Merging similar nodes (threshold={similarity_threshold})...")
    merger = EmbeddingBasedNodeMerger(
        similarity_threshold=similarity_threshold,
        embedder_path=embedder_path,
        candidate_finder_strategy=candidate_finder_strategy,
        ann_k=ann_k,
    )

    start_time = time.time()
    kg.merge_similar_nodes(merger)
    log_timing("Node merging", start_time)
    logger.info(f"✅ Graph after merging: {kg}")
    logger.debug("Merged Graph details:")
    logger.debug(kg)

    # Save the KnowledgeGraph
    log_section("SAVING GRAPH")
    logger.info(f"💾 Saving knowledge graph to {output}...")
    start_time = time.time()
    kg.save_to_file(output)
    log_timing("Graph saving", start_time)
    logger.info(f"✅ Knowledge graph successfully saved to {output}")


@cli.command("subgraph")
@click.option(
    "--graph-path",
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
    help="Path to the knowledge graph file. If not provided, uses the default example data.",
)
@click.option(
    "--embedder-path",
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
    help="Path to the embedder file. If not provided, it computes the embeddings on the fly.",
)
@click.option(
    "--num-subgraphs",
    type=int,
    default=10,
    help="Number of subgraphs to generate.",
)
@click.option(
    "--max-nodes",
    type=int,
    default=20,
    help="Maximum number of nodes to include in the subgraph.",
)
@click.option(
    "--max-steps",
    type=int,
    default=10,
    help="Maximum number of steps to take in the random walk.",
)
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, dir_okay=True),
    default="output",
    help="Output directory for saving subgraphs.",
)
@click.option(
    "--method",
    type=click.Choice(
        [
            "random_walk",
            "llm_walk",
            "embedding_path",
            "randomized_embedding_path",
            "shortest_path",
        ]
    ),
    default="random_walk",
    help="Method to use for subgraph extraction (default: random_walk).",
)
@click.option(
    "--min-score",
    type=float,
    default=4,
    help="Minimum score for a subgraph to be considered valid (default: 4).",
)
@click.option(
    "--neighbor-probability",
    type=float,
    default=0.5,
    help="Probability of selecting a neighbor node (default: 0.5).",
)
@click.option(
    "--llm",
    type=str,
    default="small",
    help="LLM to use for subgraph generation.",
)
def extract_subgraph(
    graph_path,
    embedder_path,
    num_subgraphs,
    max_nodes,
    max_steps,
    output_dir,
    method,
    min_score,
    neighbor_probability,
    llm,
):
    """Extract subgraphs from a knowledge graph."""
    log_section("SUBGRAPH EXTRACTION")
    logger.info("📊 Configuration:")
    logger.info(f"   Method: {method}")
    logger.info(f"   Number of subgraphs: {num_subgraphs}")
    logger.info(f"   Max nodes: {max_nodes}")
    logger.info(f"   Max steps: {max_steps}")
    logger.info(f"   Minimum score: {min_score}")
    logger.info(f"   Neighbor probability: {neighbor_probability}")
    logger.info(f"   LLM: {llm}")

    # Determine graph path
    if graph_path:
        graph_path = Path(graph_path)
        logger.info(f"📂 Using provided knowledge graph: {graph_path}")
    else:
        graph_path = (
            Path(__file__).parent.parent.parent.parent / "data" / "knowledge_graph.pkl"
        )
        logger.info(f"📂 Using default example knowledge graph: {graph_path}")

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
    kg = KnowledgeGraph.load_from_file(graph_path)
    log_timing("Knowledge graph loading", start_time)
    logger.info("✅ Loaded graph successfully")
    logger.debug("Knowledge Graph details:")
    logger.debug(kg)

    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"📁 Output directory: {output_path}")

    # Generate subgraphs
    for i in range(num_subgraphs):
        log_section(f"GENERATING SUBGRAPH {i + 1}/{num_subgraphs}")
        subgraph = None
        attempt = 0

        while True:
            attempt += 1
            try:
                logger.info(
                    f"🔄 Attempt {attempt}: Generating subgraph with method '{method}'"
                )
                start_time = time.time()
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
                        method=LLMWalkGenerator(max_steps=max_steps, llm="small"),
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
                    logger.debug(f"Subgraph details: {subgraph}")
                    logger.debug(f"Justification: {subgraph._path_score_justification}")
                    logger.warning("🔄 Retrying...")
                    continue
                else:
                    logger.info(f"✅ {score_info} - Above minimum threshold")
                    logger.info("🔄 Contextualizing subgraph...")
                    start_time = time.time()
                    subgraph.contextualize(llm=get_llm(llm))
                    log_timing("Contextualization", start_time)
                    break
            except Exception as e:
                logger.error(f"❌ Error in attempt {attempt}: {e}")
                if attempt >= 3:
                    logger.warning(
                        f"⚠️ Failed after {attempt} attempts, moving to next subgraph"
                    )
                    break
                logger.info("🔄 Retrying...")

        # Save the subgraph
        if subgraph:
            subdir = output_path

            subdir.mkdir(parents=True, exist_ok=True)
            output_name = f"{subgraph.start_node}_{subgraph.end_node}".replace("/", "_")
            output_file = subdir / output_name
            subgraph.save_to_file(output_file.with_suffix(".subgraph.json"))
            logger.info(f"✅ Saved to {output_file}")
        else:
            logger.warning(f"⚠️ No valid subgraph generated for iteration {i + 1}")

    log_section("PROCESS COMPLETED")
    logger.info(f"✅ Generated {num_subgraphs} subgraphs")


@cli.command("graph-stats")
@click.option(
    "--graph-path",
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
    help="Path to the knowledge graph file. If not provided, uses the default example data.",
)
def graph_stats(
    graph_path,
):
    """Display statistics about a knowledge graph."""
    log_section("GRAPH STATISTICS")

    # Determine graph path
    if graph_path:
        graph_path = Path(graph_path)
        logger.info(f"📂 Using provided knowledge graph: {graph_path}")
    else:
        graph_path = (
            Path(__file__).parent.parent.parent.parent / "data" / "knowledge_graph.pkl"
        )
        logger.info(f"📂 Using default example knowledge graph: {graph_path}")

    # Load knowledge graph
    logger.info("🔄 Loading knowledge graph...")
    start_time = time.time()
    kg = KnowledgeGraph.load_from_file(graph_path)
    log_timing("Knowledge graph loading", start_time)
    logger.info("✅ Loaded graph successfully")

    # Display graph statistics
    logger.info(kg)
    logger.info("📊 Graph Statistics:")
    logger.info(f"   Number of nodes: {len(kg.get_nodes())}")
    logger.info(f"   Number of edges: {len(kg.get_edges())}")
    logger.info(
        f"   Average node degree: {len(kg.get_edges()) / len(kg.get_nodes()):.2f}"
    )

    log_section("PROCESS COMPLETED")
    logger.info("✅ Graph statistics displayed successfully")


if __name__ == "__main__":
    cli()
