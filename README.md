# ardcore: Autonomous Research Discovery (ARD)

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://choosealicense.com/licenses/mit/)
[![Code style: ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

ARD is a Python framework for building, curating, and mining knowledge graphs to accelerate scientific progress. It provides a complete, end-to-end pipeline to transform raw data into novel, AI-generated research hypotheses.


## ✨ Features

The ARD framework is built around a three-stage pipeline, supported by a set of powerful core features.

#### Pipeline Stages
1. **Knowledge Graph Construction**: Automatically process raw documents to extract structured triplets and build a comprehensive, queryable knowledge graph.
2. **Intelligent Subgraph Extraction**: Go beyond simple queries. Use various methods—including random walks, shortest path, and LLM-guided exploration—to mine your graph for meaningful conceptual connections.
3. **Agent-Driven Hypothesis Generation**: The pinnacle of the pipeline. A sophisticated multi-agent system, built with LangGraph, analyzes subgraphs to reason about scientific connections and formulate novel, testable hypotheses.

#### Core Framework
- **Powerful CLI**: Access the core functionality of the pipeline through a simple and robust command-line interface.
- **Multiple Subgraph Strategies**: Choose from a variety of subgraph generation algorithms, from simple random walks to sophisticated LLM-guided exploration, to find the most relevant insights in your data.
- **Embedding-based Intelligence**: Leverages modern sentence-transformer and LLM embeddings for core tasks like semantic node merging and intelligent pathfinding.

## 📦 Installation

ARD requires Python 3.12+ and uses [UV](https://github.com/astral-sh/uv) as its package manager.

```bash
# Clone the repository
git clone https://github.com/your-username/ardcore.git
cd ardcore

# Create and activate a virtual environment
uv venv
source .venv/bin/activate

# Install dependencies into the virtual environment
uv pip install -e .
```

### Configuration

To use features that rely on external services, such as the LLM-powered examples, you will need to provide API keys. 

Create a `.env` file in the root of the project (you can use `.env.example` as a template) and add the necessary keys (e.g., `OPENAI_API_KEY`, `LANGFUSE_SECRET_KEY`).

## ⚙️ The ARD Pipeline

The ARD pipeline is composed of three primary stages. You can run each stage easily from the command line for a quick demonstration, or run the example Python scripts to see how you can use `ard` as a library in your own code.

---

### Step 1: Building a Knowledge Graph

**What it does:** This stage processes a dataset of documents (e.g., research papers), extracts structured `subject-predicate-object` triplets, and assembles them into a unified knowledge graph.

**Quickstart via CLI:**
The easiest way to get started is with the `ard graph` command. This single command will build a knowledge graph from the included sample data.
```bash
ard graph \
    --data-path examples/full_graph_example/data_initial/ \
    --output examples/full_graph_example/output_script/final_graph.json \
    --kg-version baseline_1
```

**For Developers: Run the Example**
To understand how to build a graph programmatically, run the full example script. This is the best way to see how to integrate `ard` into your own applications.
```bash
python examples/full_graph_example/example.py
```
This script processes the sample data in `examples/full_graph_example/data_initial` and `data_incremental`, saving the final graph to `examples/full_graph_example/output_script/final_graph.json`.

---

### Step 2: Generating Subgraphs

**What it does:** Once you have a knowledge graph, this stage intelligently explores it to find smaller, conceptually interesting subgraphs. These subgraphs form the basis for hypothesis generation.

**Quickstart via CLI:**
Use the `ard subgraph` command to extract a subgraph from the graph you built in Step 1. You can experiment with different extraction methods.
```bash
ard subgraph \
    --graph-path examples/full_graph_example/output_script/final_graph.json \
    --output-dir examples/generate_subgraph/output/ \
    --method llm_walk \
    --num-subgraphs 1
```

**For Developers: Run the Example**
To see the code behind subgraph generation, run the example script. This will load the graph from Step 1 and use an LLM-guided walk to find and save a new subgraph.
```bash
python examples/generate_subgraph/example.py
```
---

### Step 3: Generating Hypotheses

**What it does:** This is the final and most advanced stage. It takes a subgraph and uses a multi-agent system to analyze it, reason about the connections, and formulate a novel scientific hypothesis.

**Quickstart via CLI:**
Due to its complexity, this stage is best run via the example script, which is pre-configured to work with the output from Step 2.

**For Developers: Run the Example**
This script loads a subgraph from the output of Step 2, runs it through the agentic workflow, and saves the resulting hypothesis.
```bash
python examples/generate_hypothesis/example.py
```
> **Note:** This example uses Langfuse for tracing the agentic workflow. You may need to configure your Langfuse credentials to see the full traces.

## 🏗️ Architecture

The core logic of the `ard` library resides in the `src/ard/` directory:

- **`data/`**: Handles data ingestion, processing, and triplet extraction.
- **`knowledge_graph/`**: The core knowledge graph implementation, including node merging.
- **`subgraph/`**: Tools for subgraph extraction and analysis.
- **`hypothesis/`**: Classes for representing and saving hypotheses.
- **`storage/`**: Abstractions for handling local and S3 file storage.
- **`cli.py`**: Defines the command-line interface.


## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
