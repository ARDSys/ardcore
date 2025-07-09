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
    """You are a sophisticated ontologist and an expert in {scientific_domain} research.

    Given the following key concepts extracted from a comprehensive knowledge graph, your task is to evaluate the scientific quality of relationships between nodes in this subgraph. Rate the subgraph on a scale from 1 to 5, where:
    
    1: Very poor - The relationships are trivial, incorrect, or have no meaningful scientific value for {scientific_domain} research.
    2: Poor - The relationships have minimal scientific relevance or contain major misconceptions about {scientific_domain} mechanisms or concepts.
    3: Adequate - The relationships have some scientific merit, but may lack depth or novelty in the context of {scientific_domain} research.
    4: Good - The relationships represent meaningful scientific connections that contribute to understanding {scientific_domain} mechanisms or advancing the field.
    5: Excellent - The relationships reveal insightful, non-obvious scientific connections that could meaningfully advance {scientific_domain} research and provide novel insights.
    
    Provide your numerical rating and a brief justification for your assessment based on scientific validity, relevance to {scientific_domain}, and potential research impact.
    
    Answer in the following format:
    rating=<rating>
    <justification>

    Consider this list of nodes and relationships from a knowledge graph between "{start_node}" and "{end_node}". 

    The format of the knowledge graph is "(node_1)-[:relationship between node_1 and node_2]->(node_2),\n(node_2)-[:relationship between node_2 and node_3]->(node_3)..."

    Here is the graph:

    {graph_str}
    """
)
