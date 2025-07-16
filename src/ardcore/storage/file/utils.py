
import tiktoken

SUBGRAPH_FILE_EXTENSION = ".subgraph.json"


def get_subgraph_file_name(subgraph_name: str) -> str:
    return f"{subgraph_name}{SUBGRAPH_FILE_EXTENSION}"


def get_subgraph_name(subgraph_path: str) -> str:
    subgraph_filename = subgraph_path.split("/")[-1]
    return subgraph_filename.split(SUBGRAPH_FILE_EXTENSION)[0]


def count_tokens(text: str, model: str = "gpt-4o") -> int:
    """
    Count the number of tokens in the given text using tiktoken for the specified model.
    Args:
        text (str): The text to tokenize.
        model (str): The model identifier to use for tokenization (default is "gpt-4o").
    Returns:
        int: The total number of tokens.
    """
    encoder = tiktoken.encoding_for_model(model)
    return len(encoder.encode(text))


def merge_punctuation_chunks(chunks: list[str]) -> list[str]:
    """
    Merge any text chunk that starts with punctuation into its preceding chunk.
    This function checks each chunk in the list. If a chunk starts with a punctuation
    character ('.', '?', or '!'), it appends that punctuation to the previous chunk and
    removes it from the start of the current chunk.
    Args:
        chunks (List[str]): A list of text chunks.
    Returns:
        List[str]: The list of text chunks with leading punctuation merged into the previous chunk.
    """
    merged_chunks: list[str] = []
    for chunk in chunks:
        if merged_chunks and chunk and chunk[0] in ".?!":
            # Append the punctuation to the previous chunk and remove it from the current chunk.
            merged_chunks[-1] = merged_chunks[-1].rstrip() + chunk[0]
            chunk = chunk[1:].lstrip()
        merged_chunks.append(chunk)
    return merged_chunks
