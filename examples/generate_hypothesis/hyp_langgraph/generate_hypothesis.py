import json
from pathlib import Path

import click
import dotenv
from langchain_core.runnables import RunnableConfig
from langfuse.callback import CallbackHandler
from loguru import logger

from ardcore.subgraph import Subgraph

from .graph import hypgen_graph
from .utils import message_to_dict

langfuse_callback = CallbackHandler()

dotenv.load_dotenv()


@click.command()
@click.option(
    "--file", "-f", type=click.Path(exists=True), help="Path to the json file"
)
@click.option(
    "--output",
    "-o",
    type=click.Path(exists=True, file_okay=False),
    help="Path to the output directory",
    default=".",
)
def main(file: str, output: str):
    file_path = Path(file)
    output_path = Path(output)
    subgraph = Subgraph.load_from_file(file_path)

    logger.info(f"Subgraph loaded from {file_path}: {subgraph}")

    context = subgraph.context
    path = subgraph.to_cypher_string(full_graph=False)

    res = hypgen_graph.invoke(
        {"subgraph": path, "context": context},
        config=RunnableConfig(callbacks=[langfuse_callback], recursion_limit=100),
    )
    logger.info(f"Hypothesis generated for {file_path}")

    output_file = output_path / file_path.name.replace(
        "subgraph.json", "hypothesis.json"
    )
    output_file.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        res["messages"] = [message_to_dict(message) for message in res["messages"]]
        f.write(json.dumps(res, indent=4))
        logger.info(f"Hypothesis saved to {output_file}")


if __name__ == "__main__":
    main()
