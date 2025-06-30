# ARD CLI Tool

A command-line interface for ARD's knowledge graph and subgraph operations.

## Installation

Install the package:

```bash
pip install -e .
```

This will make the `ard-cli` command available in your environment.

## Usage

The CLI tool provides two main commands:

### 1. Creating Knowledge Graphs

```bash
ard-cli graph --data-path /path/to/data --output knowledge_graph.pkl --max-items 100 --similarity-threshold 0.85
```

Options:
- `--data-path`: Path to the data directory (optional, uses default example data if not provided)
- `--output`: Path to save the knowledge graph (default: knowledge_graph.pkl)
- `--max-items`: Maximum number of items to process (default: 10)
- `--similarity-threshold`: Similarity threshold for merging nodes (default: 0.85)
- `--kg-version`: The version of KG (e.g., baseline_1) to continue running

### 2. Extracting Subgraphs and Generating Tweets

```bash
ard-cli subgraph --graph-path knowledge_graph.pkl --output-dir output --num-tweets 5 --persona mad_scientist --method two_nodes
```

Options:
- `--graph-path`: Path to the knowledge graph file (optional, uses default if not provided)
- `--embedder-path`: Path to the embedder file (optional)
- `--persona`: Persona to use for tweet generation (default: mad_scientist)
- `--num-tweets`: Number of tweets to generate (default: 10)
- `--max-nodes`: Maximum number of nodes in the subgraph (default: 20)
- `--max-steps`: Maximum number of steps in random walk (default: 10)
- `--output-dir`: Output directory for saving subgraphs and tweets (default: output)
- `--method`: Method to use for subgraph extraction [random_walk|two_nodes] (default: two_nodes)

## Examples

Create a knowledge graph from the default example data:

```bash
ard-cli graph --output my_kg.pkl
```

Extract subgraphs and generate tweets using the random walk method:

```bash
ard-cli subgraph --graph-path my_kg.pkl --method random_walk --num-tweets 3
```

## Help

Get help on available commands:

```bash
ard-cli --help
```

Get help on a specific command:

```bash
ard-cli graph --help
ard-cli subgraph --help
``` 
