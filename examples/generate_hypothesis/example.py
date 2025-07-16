import glob
import hashlib
import json
import time
from pathlib import Path

import dotenv
from hyp_langgraph.graph import create_hypgen_graph
from hyp_langgraph.utils import calculate_message_cost, message_to_dict
from langchain_core.runnables import RunnableConfig
from langfuse.callback import CallbackHandler
from loguru import logger

from ardcore.subgraph import Subgraph

# --- Configuration ---
BASE_DIR = Path(__file__).parent
INPUT_SUBGRAPH_DIR = BASE_DIR.parent / "generate_subgraph/output"
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Initialize langfuse callback
langfuse_callback = CallbackHandler()

# Load environment variables
dotenv.load_dotenv()


def log_timing(operation_name, start_time):
    """Log the time taken for an operation in a consistent format."""
    elapsed = time.time() - start_time
    logger.info(f"⏱️ {operation_name} completed in {elapsed:.2f} seconds")


def log_section(section_name):
    """Create a visual separator for a new section in logs."""
    logger.info(f"\n{'=' * 50}")
    logger.info(f"📌 {section_name}")
    logger.info(f"{'=' * 50}")


def save_hypothesis(
    hypothesis: dict,
    subgraph: Subgraph,
):
    """Save hypothesis results to a local file."""
    # Normalize messages before saving
    hypothesis["messages"] = [
        message_to_dict(message) for message in hypothesis["messages"]
    ]
    hypothesis["total_cost"] = sum(
        calculate_message_cost(message) for message in hypothesis["messages"]
    )
    hypothesis["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
    hypothesis["subgraph_id"] = subgraph.subgraph_id
    hypothesis["hypothesis_id"] = hashlib.sha256(hypothesis["hypothesis"].encode()).hexdigest()

    hypothesis_id = hypothesis["hypothesis_id"]
    output_file = OUTPUT_DIR / f"{hypothesis_id}.json"

    logger.info(f"💾 Saving hypothesis to {output_file}")

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(hypothesis, f, indent=2, ensure_ascii=False)

    logger.info("✅ Hypothesis saved successfully")


def generate_hypothesis(
    subgraph_path: str,
    scientific_domain: str,
):
    """Generate hypothesis from a subgraph."""
    log_section("HYPOTHESIS GENERATION")

    subgraph_data = Subgraph.load_from_file(
        filename=subgraph_path,
        scientific_domain=scientific_domain,
    )

    log_section("GENERATING HYPOTHESIS")
    start_time = time.time()

    context = subgraph_data.context
    subgraph_cypher = subgraph_data.to_cypher_string(full_graph=False)

    hypgen_graph = create_hypgen_graph()
    res = hypgen_graph.invoke(
        {"subgraph": subgraph_cypher, "context": context},
        config=RunnableConfig(callbacks=[langfuse_callback], recursion_limit=100),
    )

    log_timing("Hypothesis generation", start_time)

    save_hypothesis(
        hypothesis=res,
        subgraph=subgraph_data,
    )
    return res


def main():
    # Find the first subgraph file in the input directory
    subgraph_files = glob.glob(
        str(INPUT_SUBGRAPH_DIR / "**/*.subgraph.json"), recursive=True
    )
    if not subgraph_files:
        logger.error(
            f"No subgraphs found in {INPUT_SUBGRAPH_DIR}. Please run the 'generate_subgraph' example first."
        )
        return

    subgraph_path = subgraph_files[0]
    logger.info(f"Using subgraph: {subgraph_path}")

    generate_hypothesis(
        subgraph_path=subgraph_path,
        scientific_domain="example_scientific_domain",
    )


if __name__ == "__main__":
    main()
