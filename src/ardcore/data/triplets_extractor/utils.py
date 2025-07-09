import json
import re
from typing import List

import litellm
from langfuse.decorators import langfuse_context, observe
from litellm import completion
from pydantic import BaseModel

from ardcore.data.triplets import Triplets

litellm.enable_json_schema_validation = True


def triplets_to_str(triplets: Triplets):
    triplets_str = [f"({t.node_1})-[{t.edge}]->({t.node_2})" for t in triplets.triplets]
    triplets_str = "\n".join(triplets_str)
    return triplets_str


# set callbacks
litellm.success_callback = ["langfuse"]
litellm.failure_callback = ["langfuse"]


@observe()
def generate_response(model_name, sys_message, user_prompt):
    messages = [
        {"content": sys_message, "role": "system"},
        {"content": user_prompt, "role": "user"},
    ]
    response = completion(
        model=model_name,
        messages=messages,
        metadata={
            "existing_trace_id": langfuse_context.get_current_trace_id(),  # set langfuse trace ID
            "parent_observation_id": langfuse_context.get_current_observation_id(),
        },
    )
    return response.choices[0].message.content


def find_triplets_in_text(text, triplets, model_name):
    PROMPT = f"""
You are a meticulous information extractor.
Your task is to find factual evidence in a given article that supports a set of triplets.
For each triplet, return one verbatim snippet from the article that directly supports the triplet.
If no exact support is found, return "NULL".
You must not hallucinate or infer. Only return snippets that are explicitly grounded in the article.
Input

Triplets:
{triplets}

Article:
{text}

Output Format

Return the extracted triplets as a JSON-formatted list of triplet dictionaries:

    [
    {{"node_1": "subject1", "node_2": "object1", "edge": "predicate1", "snippet": "snippet1" }},
    {{"node_1": "subject2", "node_2": "object2", "edge": "predicate2", "snippet": "snippet2" }},
    ...
    ]

Rules

    Return only exact matches based on the article.

    The snippet can be one or two full sentences max.

    No rephrasing or summarizing. Use original text.

    Be conservative: only assign a snippet if it clearly supports the triplet.
    """
    response = generate_response(model_name, PROMPT, text)

    # Parse the response to get the triplets
    try:
        # Extract the JSON part from the response (in case there's surrounding text)
        json_match = re.search(r"\[\s*{.*}\s*\]", response, re.DOTALL)
        if json_match:
            triplets_json = json_match.group(0)
        else:
            triplets_json = response

        # Parse the JSON
        triplets = json.loads(triplets_json)
        return triplets
    except json.JSONDecodeError as e:
        print(f"Error parsing triplets JSON: {e}")
        print(f"Response was: {response}")
        return []


PROMPT = """
# TASK
You are given a list of subject–relation–object triplets extracted from scientific publications using LLM-based, open generative relation extraction methods. These triplets originate from multiple local knowledge graphs (KGs) and may use inconsistent entity naming, varied relation phrasing, and lack a unified schema or ontology. Your goal is to induce a bottom-up ontology/schema and produce a harmonized set of triplets that can be merged into a single, global KG for the domain of {scientific_domain}.

# RESPONSIBILITIES
## Entity Resolution & Normalization:


Identify and merge references to the same real-world entities (people, concepts, proteins, etc.), resolving ambiguity and redundancy (e.g., "Dr. John Smith", "J. Smith", "Smith, John D." → a canonical form).
For each entity, choose a consistent, unambiguous name or identifier; if uncertain, prefer the most complete or widely-used variant.

## Relation Standardization:


Group and unify semantically equivalent relations (e.g., "discovered", "identified", "found", "reported" → "discovered").
Define a standard set of relation types as part of your lightweight schema; for each triplet, replace the original relation with the standardized one.

## Implicit Ontology Induction:


Infer and specify a lightweight, shared schema or ontology from the input triplets, capturing common entity types, relation types, and any useful hierarchies or constraints.
Explicitly list the schema/ontology you have inferred (as a list of entity types and relation types) before returning the harmonized triplets.

## Artifact & Noise Filtering:


Detect and filter out triplets that are likely hallucinated, inconsistent, or obviously erroneous based on redundancy, contradiction, or lack of plausibility.

# OUTPUT FORMAT
First, summarize the induced schema/ontology (list entity types, relation types, and brief definitions).
Second, return the cleaned and standardized triplets following your induced structure.

---

# INPUT TRIPLETS
{triplets_here}

---

Please process the above triplets following the instructions above.
"""


class Triplet(BaseModel):
    node_1: str
    edge: str
    node_2: str


class RefinedTriplets(BaseModel):
    triplets: List[Triplet]


def refine_triplets_with_llm(
    triplets: Triplets, scientific_domain: str, model: str = "gpt-4o-mini"
):
    triplets_str = triplets_to_str(triplets)
    response = litellm.completion(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": PROMPT.format(
                    scientific_domain=scientific_domain, triplets_here=triplets_str
                ),
            }
        ],
        response_format=RefinedTriplets,
    )
    refined_triplets = RefinedTriplets.model_validate_json(
        response.choices[0].message.content
    )
    return refined_triplets
