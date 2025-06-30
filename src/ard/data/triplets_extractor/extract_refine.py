import json
import re
import time

from loguru import logger

from ard.data.triplets_extractor.utils import generate_response


def extract_refine_generator(context, config):
    """
    Generate a local knowledge graph from a context.

    Args:
        context (str): The raw context text
        extractor_model_name (str): The name of the LLM to use for extraction
        refiner_model_name (str): The name of the LLM to use for refinement
        repeat_refine (int): Number of times to refine the triplets

    Returns:
        tuple: (nx.DiGraph, list) - The local graph and the refined triplets
    """
    logger.debug(f"Context length: {len(context)} characters")

    extractor_model_name = config["extractor_model_name"]
    refiner_model_name = config["refiner_model_name"]
    refiner_model_sleep = config.get("refiner_model_sleep", 0)
    repeat_refine = config["max_iterations"]

    # Extract initial triplets
    start_time = time.time()
    triplets = extract_triples_from_context(context, extractor_model_name)
    logger.debug(
        f"Extracted {len(triplets)} initial triplets in {time.time() - start_time:.2f} seconds"
    )

    # Refine triplets if requested
    refined_triplets = triplets
    for i in range(repeat_refine):
        logger.debug(f"Refining triplets (iteration {i + 1}/{repeat_refine})")
        start_time = time.time()
        if refiner_model_sleep > 0:
            time.sleep(refiner_model_sleep)
        refined_triplets = refine_triples(context, refined_triplets, refiner_model_name)
        logger.debug(f"Refined triplets in {time.time() - start_time:.2f} seconds")

    # Create the local graph
    # local_graph = create_local_graph(refined_triplets, chunk_id)

    # logger.info(f"Generated local graph with {local_graph.number_of_nodes()} nodes and {local_graph.number_of_edges()} edges")

    return refined_triplets


def refine_triples(context, initial_triplets, model_name):
    """
    Refine the initial triplets to have consistent labels using the LLM.

    Args:
        context (str): The raw context text
        initial_triplets (list): List of initial triplets
        model_name (str): The name of the LLM to use

    Returns:
        list: List of refined triplets
    """
    # Convert the initial triplets to a JSON string
    initial_triplets_json = json.dumps(initial_triplets, indent=2)

    system_prompt = (
        "You are a knowledge graph expert who refines and improves ontology triplets. "
        "Your task is to review the existing triplets and make them more consistent and accurate."
    )

    user_prompt = (
        f"Read this context: ```{context}```\n"
        f"Read this ontology: ```{initial_triplets_json}```\n\n"
        f"Improve the ontology by renaming nodes so that they have consistent labels that are widely used in the field of materials science. "
        f'Make sure the output follows the exact same JSON format as the input, with keys "node_1", "node_2", and "edge".'
    )

    # Generate the refined triplets using the LLM
    response = generate_response(
        model_name=model_name, sys_message=system_prompt, user_prompt=user_prompt
    )

    # Parse the response to get the refined triplets
    try:
        # Extract the JSON part from the response (in case there's surrounding text)
        json_match = re.search(r"\[\s*{.*}\s*\]", response, re.DOTALL)
        if json_match:
            triplets_json = json_match.group(0)
        else:
            triplets_json = response

        # Parse the JSON
        refined_triplets = json.loads(triplets_json)
        return refined_triplets
    except json.JSONDecodeError as e:
        print(f"Error parsing refined triplets JSON: {e}")
        print(f"Response was: {response}")
        return initial_triplets  # Return the initial triplets if parsing fails


def extract_triples_from_context(context, model_name):
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
    SYS_PROMPT_GRAPHMAKER = (
        "You are a network ontology graph maker who extracts terms and their relations from a given context, using category theory. "
        "You are provided with a context chunk (delimited by ```) Your task is to extract the ontology "
        "of terms mentioned in the given context. These terms should represent the key concepts as per the context, including well-defined and widely used names of materials, systems, methods. \n\n"
        "Format your output as a list of JSON. Each element of the list contains a pair of terms"
        "and the relation between them, like the following: \n"
        "[\n"
        "   {\n"
        '       "node_1": "A concept from extracted ontology",\n'
        '       "node_2": "A related concept from extracted ontology",\n'
        '       "edge": "Relationship between the two concepts, node_1 and node_2, succinctly described"\n'
        "   }, {...}\n"
        "]\n"
        ""
        "Examples:"
        "Context: ```Alice is Marc's mother.```\n"
        "[\n"
        "   {\n"
        '       "node_1": "Alice",\n'
        '       "node_2": "Marc",\n'
        '       "edge": "is mother of"\n'
        "   }"
        "]\n"
        "Context: ```Silk is a strong natural fiber used to catch prey in a web. Beta-sheets control its strength.```\n"
        "[\n"
        "   {\n"
        '       "node_1": "silk",\n'
        '       "node_2": "fiber",\n'
        '       "edge": "is"\n'
        "   },"
        "   {\n"
        '       "node_1": "beta-sheets",\n'
        '       "node_2": "strength",\n'
        '       "edge": "control"\n'
        "   },"
        "   {\n"
        '       "node_1": "silk",\n'
        '       "node_2": "prey",\n'
        '       "edge": "catches"\n'
        "   }"
        "]\n\n"
        "Analyze the text carefully and produce around 10 triplets, making sure they reflect consistent ontologies.\n"
    )

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
