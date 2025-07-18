import json
import re
import time

from loguru import logger

from ardcore.data.triplets_extractor.utils import generate_response


def extract_review_generator(context, config):
    """
    Generate a local knowledge graph from a context.

    Args:
        context (str): The raw context text
        model_name (str): The name of the LLM to use
        repeat_refine (int): Number of times to refine the triplets

    Returns:
        tuple: (nx.DiGraph, list) - The local graph and the refined triplets
    """
    logger.debug(f"Context length: {len(context)} characters")

    extractor_model_name = config["extractor_model_name"]
    reviewer_model_name = config["reviewer_model_name"]
    reviewer_model_sleep = config.get("reviewer_model_sleep", 0)
    max_iterations = config["max_iterations"]

    # Extract initial triplets
    start_time = time.time()

    triplets = []
    review = ""

    for i in range(max_iterations):
        logger.debug(f"Extracting triplets (iteration {i + 1}/{max_iterations})")
        start_time = time.time()
        triplets = extract_triples_from_context(
            context, triplets, review, extractor_model_name
        )
        logger.debug(
            f"Extracted {len(triplets)} triplets in {time.time() - start_time:.2f} seconds"
        )
        if reviewer_model_sleep > 0:
            logger.debug(
                f"Sleeping for {reviewer_model_sleep} seconds before reviewing triplets"
            )
            time.sleep(reviewer_model_sleep)
        logger.debug("Reviewing triplets")
        review = review_triples(context, triplets, reviewer_model_name)
        logger.debug(f"Reviewed triplets in {time.time() - start_time:.2f} seconds")
        if review.upper().endswith("ACCEPT"):
            break

    if not review.upper().endswith("ACCEPT"):
        logger.warning(f"Did not accept triplets after {max_iterations} iterations")

    return triplets


def review_triples(context, triplets, model_name):
    """
    Review the initial triplets to have consistent labels using the LLM.

    Args:
        context (str): The raw context text
        initial_triplets (list): List of initial triplets
        model_name (str): The name of the LLM to use

    Returns:
        list: List of refined triplets
    """
    # Convert the initial triplets to a JSON string
    triplets_json = json.dumps(triplets, indent=2)

    system_prompt = """
    The following triplets were generated by a language model using open generative relation extraction (GRE) from a scientific paper (included below) in the field of rheumatology. These triplets will be integrated into a Knowledge Graph designed to support a multi-agent system in generating novel, AI-driven scientific hypotheses.

Please carefully evaluate each extracted triplet using the following metrics:

1. Topical Similarity (Relevance): Do the triplets accurately and comprehensively reflect the primary topics discussed in the original text? Are there any significant topics from the source text not represented adequately by the extracted triplets?

2. Uniqueness (Diversity): Assess whether the extracted triplets are diverse and distinct. Identify any triplets that appear overly similar or redundant, potentially diminishing the breadth of captured information.

3. Factualness (Accuracy): Confirm that each triplet accurately reflects the factual content of the original text. Highlight any triplets containing information not supported by the source material or representing potential model hallucinations.

4. Granularity (Detail Level): Determine whether each triplet captures information at an appropriate level of specificity. Indicate triplets that are overly general or could be subdivided into more precise and detailed statements.

5. Completeness (Coverage): Evaluate the overall completeness of the extracted information relative to the source document. If known reference ('gold standard') triplets are available, please note whether critical information from these is missing from the extracted set.

Provide constructive feedback clearly identifying any issues related to relevance, diversity, factualness, granularity, or completeness. Suggest specific improvements or corrections to help enhance the quality of the relation extraction output. If the triplets are good and no feedback is needed respond with "ACCEPT".
    """

    user_prompt = (
        f"Original Text: {context}\nExtracted triplets: {triplets_json}\nReview:"
    )

    # Generate the refined triplets using the LLM
    response = generate_response(
        model_name=model_name, sys_message=system_prompt, user_prompt=user_prompt
    )

    return response


def extract_triples_from_context(context, triplets, review, model_name):
    """
    Extract triples (node_1, edge, node_2) from a given context using an LLM.

    Args:
        context (str): The raw context text
        generate (function): Function to call the LLM
        temperature (float): Temperature for generation
        max_tokens (int): Maximum tokens for response

    Returns:
        list: List of dictionaries with node_1, node_2, and edge
    """

    SYS_PROMPT_GRAPHMAKER = """
    Extract clear and concise triplets from the provided text using open generative relation extraction (GRE). 
    Each triplet must precisely represent a factual relationship from the text and be formatted as (subject, predicate, object).
    Ensure each triplet is accurate, relevant, specific, and non-redundant.

    If provided, use previously generated triplets together with their review to improve the quality of the triplets.

    Return the extracted triplets as a JSON-formatted list of triplet dictionaries:

    [
    { "node_1": "subject1", "node_2": "object1", "edge": "predicate1" },
    { "node_1": "subject2", "node_2": "object2", "edge": "predicate2" },
    ...
    ]
"""

    if len(review) > 0:
        user_prompt = f"Context: ```{context}``` \n\nPrevious triplets: {triplets} \n\nPrevious review: {review} \n\nOutput: "
    else:
        user_prompt = f"Context: ```{context}``` \n\nOutput: "

    # Generate the triplets using the LLM
    response = generate_response(
        model_name=model_name,
        sys_message=SYS_PROMPT_GRAPHMAKER,
        user_prompt=user_prompt,
    )

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
