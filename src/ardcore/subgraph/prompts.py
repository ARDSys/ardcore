"""
Prompt templates for LLM interactions.
"""

from langchain_core.prompts import PromptTemplate

# Subgraph analysis prompt
SUBGRAPH_ANALYSIS_PROMPT = PromptTemplate.from_template(
    """You are a sophisticated ontologist trained in scientific research, engineering, and innovation, with particular expertise in {scientific_domain} research. 
    
Given the following key concepts extracted from a comprehensive knowledge graph, your task is to define each one of the terms and discuss the relationships identified in the graph from the perspective of {scientific_domain} research.

Consider this list of nodes and relationships from a knowledge graph between "{start_node}" and "{end_node}". 

The format of the knowledge graph is "(node_1)-[:relationship between node_1 and node_2]->(node_2),\n(node_2)-[:relationship between node_2 and node_3]->(node_3)..."

Here is the graph:

{graph_str}

Make sure to incorporate EACH of the concepts in the knowledge graph in your response. 

Do not add any introductory phrases. First, define each term in the knowledge graph in the context of {scientific_domain} research, and then, secondly, discuss each of the relationships, with context relevant to {scientific_domain}. """,
)

# Subgraph score prompt
SUBGRAPH_SCORE_PROMPT = PromptTemplate.from_template(
    """You are a creative, interdisciplinary scientific researcher and an expert in identifying breakthrough ideas in {scientific_domain} research.

    Your goal is to identify subgraphs that represent potentially groundbreaking, non-obvious connections that could spark novel scientific hypotheses. Your primary focus is on **novelty** and **plausibility**, not on whether the connections are already supported by existing literature.
    
    Rate the subgraph on a continuous scale from 1.0 to 5.0, where:
    
    1.0: **Unusable** - The relationships are factually incorrect (e.g., "Cannabis contains Psilocybin"), nonsensical (e.g., "Psilocybin has dendrites"), or trivial/redundant loops. These should be discarded.
    2.0: **Vague / Untestable** - The relationships are too generic, vague, or abstract to form a concrete, testable hypothesis (e.g., "Psilocybin describes Early life experiences").
    3.0: **Obvious / Incremental** - The relationships are scientifically valid but represent well-known facts or incremental steps in {scientific_domain} research. They lack the novelty required to spark a breakthrough.
    4.0: **Novel & Plausible** - The relationships connect concepts in a surprising way that is not well-established. There is a conceivable mechanistic or conceptual basis for the connection, suggesting a clear, testable hypothesis. **These are valuable.**
    5.0: **Breakthrough Potential** - The relationships reveal a highly insightful, non-obvious connection between disparate scientific fields. The link is mechanistically plausible and, if validated, could open an entirely new line of research or challenge the current paradigm in {scientific_domain}. **These are the highest priority.**
    
    Provide your numerical rating and a brief justification for your assessment based on **novelty, mechanistic plausibility, and potential to spark new research directions.**
    
    Use continuous scale to rate. It's not always black and white. For example - it's ok to decide on 2.9 or 3.1 instead of 3.0.
    
    Answer in the following format:
    rating=<rating>
    <justification>

    Consider this list of nodes and relationships from a knowledge graph between "{start_node}" and "{end_node}". 

    The format of the knowledge graph is "(node_1)-[:relationship between node_1 and node_2]->(node_2),\n(node_2)-[:relationship between node_2 and node_3]->(node_3)..."

    Here is the graph:

    {graph_str}
    """
)
