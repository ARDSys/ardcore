from typing import List

import litellm
from pydantic import BaseModel

from ard.data.triplets import Triplets

# litellm.enable_json_schema_validation = True


def triplets_to_str(triplets: Triplets):
    triplets_str = [f"({t.node_1})-[{t.edge}]->({t.node_2})" for t in triplets.triplets]
    triplets_str = "\n".join(triplets_str)
    return triplets_str


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
        model=model,
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
