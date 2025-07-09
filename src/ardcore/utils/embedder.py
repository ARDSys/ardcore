import json
import os
import time
from typing import Dict, List, Literal, Optional

import numpy as np
from tqdm import tqdm

# sentence_transformers will be conditionally imported
# google.generativeai will be conditionally imported


class Embedder:
    """
    A class for generating, storing, and retrieving embeddings for nodes in a knowledge graph.
    Supports different embedding providers like Gemini API or local SentenceTransformer models.
    """

    def __init__(
        self,
        embedding_provider: Literal["gemini", "sentence_transformer"] = "gemini",
        model_name: Optional[str] = None,  # Default will be set based on provider
        cache_embeddings: bool = True,
        distance_metric: str = "cosine",
        api_key: Optional[str] = None,  # Only for Gemini
        gemini_batch_size: int = 100,
    ):
        """
        Initialize the Embedder.

        Args:
            embedding_provider: The embedding provider to use ("gemini" or "sentence_transformer").
            model_name: Name of the embedding model to use.
                        For "gemini", e.g., "text-embedding-004".
                        For "sentence_transformer", e.g., "all-MiniLM-L6-v2".
                        If None, a default will be chosen based on the provider.
            cache_embeddings: Whether to cache embeddings in memory.
            distance_metric: Distance metric to use ('cosine', 'euclidean', or 'dot').
            api_key: Your Google Gemini API key (only required if provider is "gemini").
                     If None and provider is "gemini", it will try GOOGLE_API_KEY env var.
        """
        self.embedding_provider = embedding_provider
        self.cache_embeddings = cache_embeddings
        self.distance_metric = distance_metric
        self._model = None  # Lazy-loaded client or model
        self._embeddings = {}  # Cache for embeddings
        self.gemini_batch_size = gemini_batch_size

        if self.embedding_provider == "gemini":
            self.model_name = model_name or "text-embedding-004"
            self.api_key = api_key
            if not self.api_key and not os.environ.get("GOOGLE_API_KEY"):
                print(
                    "Warning: Gemini provider selected, but no API key provided directly "
                    "or via GOOGLE_API_KEY environment variable. Load will fail if API key isn't found."
                )
        elif self.embedding_provider == "sentence_transformer":
            self.model_name = model_name or "all-MiniLM-L6-v2"
            self.api_key = None  # Not used for sentence_transformer
        else:
            raise ValueError(
                f"Unsupported embedding_provider: {embedding_provider}. "
                "Choose 'gemini' or 'sentence_transformer'."
            )

    def _load_model(self):
        """
        Lazy-load the embedding model or client based on the provider.

        Raises:
            ImportError: If required packages are not installed.
            ValueError: If API key is missing for Gemini.
        """
        if self._model is None:
            if self.embedding_provider == "gemini":
                try:
                    import google.genai as genai
                except ImportError:
                    raise ImportError(
                        "The google-generativeai package is required for Gemini. "
                        "Install it with 'uv pip install google-generativeai'"
                    )

                api_key_to_use = self.api_key or os.environ.get("GOOGLE_API_KEY")
                if not api_key_to_use:
                    raise ValueError(
                        "Gemini API key not provided. Pass it to Embedder or set GOOGLE_API_KEY."
                    )
                self._model = genai.Client(api_key=api_key_to_use)

            elif self.embedding_provider == "sentence_transformer":
                try:
                    from sentence_transformers import SentenceTransformer
                except ImportError:
                    raise ImportError(
                        "The sentence-transformers package is required. "
                        "Install it with 'uv pip install sentence-transformers'"
                    )
                self._model = SentenceTransformer(self.model_name)
            else:
                # This case should ideally be caught in __init__, but as a safeguard:
                raise ValueError(
                    f"Unsupported embedding_provider: {self.embedding_provider}"
                )

    def embed(
        self,
        words: List[str],
        show_progress_bar: bool = False,  # Only for sentence_transformer
    ) -> List[np.ndarray]:
        """
        Calculate embeddings for a list of words.
        The order of returned embeddings corresponds to the order of input words.

        Args:
            words: The words to embed.
            show_progress_bar: Whether to show a progress bar (sentence_transformer only).

        Returns:
            List[np.ndarray]: List of embedding vectors, in the same order as input words.
        """
        self._load_model()

        if not words:
            return []

        if self.embedding_provider == "gemini":
            # Split words into batches of gemini_batch_size
            batches = [
                words[i : i + self.gemini_batch_size]
                for i in range(0, len(words), self.gemini_batch_size)
            ]
            embeddings_list = []
            for batch in tqdm(
                batches,
                desc="Embedding words",
                total=len(batches),
                disable=not show_progress_bar,
            ):
                for retry in range(5):
                    try:
                        api_response = self._model.models.embed_content(
                            model=self.model_name, contents=batch
                        )
                        break
                    except Exception:
                        if retry == 4:  # Last attempt
                            raise
                        time.sleep(1)
                # Ensure api_response.embeddings is a list of lists (embeddings)
                if not hasattr(api_response, "embeddings") or not isinstance(
                    api_response.embeddings, list
                ):
                    raise ValueError(
                        "Gemini API did not return expected embeddings structure (missing or not a list)."
                    )
                # Further check structure if list is not empty - this was missing and causing issues with mocks/empty returns
                if api_response.embeddings and not all(
                    hasattr(emb, "values") for emb in api_response.embeddings
                ):
                    raise ValueError(
                        "Gemini API: Elements in embeddings list do not have 'values' attribute."
                    )
                embeddings_list.extend(
                    [embedding.values for embedding in api_response.embeddings]
                )
                if len(batch) != len(api_response.embeddings):
                    raise ValueError(
                        "Mismatch between number of input words and returned Gemini embeddings."
                    )
            if len(words) != len(embeddings_list):
                raise ValueError(
                    "Mismatch between number of input words and returned Gemini embeddings."
                )
        elif self.embedding_provider == "sentence_transformer":
            embeddings_list = self._model.encode(
                words, show_progress_bar=show_progress_bar
            )
            # Ensure embeddings_list is a 2D numpy array or list of lists
            if (
                isinstance(embeddings_list, np.ndarray)
                and len(embeddings_list.shape) == 1
            ):
                # Single word case might return 1D array
                embeddings_list = embeddings_list.reshape(1, -1)
            elif (
                isinstance(embeddings_list, list)
                and embeddings_list
                and not isinstance(embeddings_list[0], (list, np.ndarray))
            ):
                # If it's a list of 1D arrays for single items, convert to list of lists
                embeddings_list = [np.asarray(e).tolist() for e in embeddings_list]

        else:
            raise ValueError(
                f"Unsupported embedding_provider: {self.embedding_provider}"
            )

        result_vectors: List[np.ndarray] = []
        for i, word in enumerate(words):
            embedding_vector = np.asarray(embeddings_list[i]).flatten()
            result_vectors.append(embedding_vector)
            if self.cache_embeddings:
                self._embeddings[word] = embedding_vector

        return result_vectors

    def get_embedding(self, text: str) -> np.ndarray:
        """
        Get the embedding for a given text.

        Args:
            text: The text to get the embedding for.
            show_progress_bar: Whether to show a progress bar (sentence_transformer only).

        Returns:
            np.ndarray: The embedding vector.
        """
        if text in self._embeddings:
            return self._embeddings[text]

        # self.embed will cache the embedding for 'text'
        embedding_vectors = self.embed([text], show_progress_bar=False)
        if not embedding_vectors:
            # This case should ideally not be reached if text is non-empty
            # and embed is expected to produce an embedding for it.
            raise ValueError(f"Embedding generation failed for text: '{text}'")
        return embedding_vectors[0]

    def get_embeddings(
        self, texts: List[str], show_progress_bar: bool = False
    ) -> Dict[str, np.ndarray]:
        """
        Get embeddings for a list of texts.

        Args:
            texts: List of texts to get embeddings for.
            show_progress_bar: Whether to show a progress bar (sentence_transformer only).

        Returns:
            Dict[str, np.ndarray]: Dictionary mapping texts to their embeddings.
        """
        if not texts:
            return {}

        results = {}
        missing_texts = []

        for text_item in texts:
            if text_item in self._embeddings:
                results[text_item] = self._embeddings[text_item]
            else:
                missing_texts.append(text_item)

        if missing_texts:
            # self.embed will cache embeddings for all items in missing_texts
            missing_embedding_vectors = self.embed(
                missing_texts, show_progress_bar=show_progress_bar
            )

            if len(missing_embedding_vectors) != len(missing_texts):
                raise ValueError(
                    "Mismatch between the number of missing texts and "
                    "the number of embeddings returned by the embed method. "
                    f"Expected {len(missing_texts)} embeddings, got {len(missing_embedding_vectors)}."
                )

            for i, text_item in enumerate(missing_texts):
                embedding_vector = missing_embedding_vectors[i]
                # The embedding is already cached by the self.embed call if cache_embeddings is True
                results[text_item] = embedding_vector
        return results

    @property
    def embeddings_len(self) -> int:
        return len(self._embeddings)

    def calculate_distance(self, text1: str, text2: str, metric: str = None) -> float:
        """
        Calculate the distance between two texts.

        Args:
            text1: First text
            text2: Second text
            metric: Distance metric to use ('cosine', 'euclidean', or 'dot'). If None,
                   uses the metric specified during initialization.

        Returns:
            float: Distance between the two texts

        Raises:
            ValueError: If the specified metric is not supported
        """
        # Get embeddings
        embedding1 = self.get_embedding(text1)
        embedding2 = self.get_embedding(text2)

        # Use provided metric or fall back to default
        metric = metric or self.distance_metric

        # Calculate distance based on selected metric
        if metric == "cosine":
            return 1.0 - self._cosine_similarity(embedding1, embedding2)
        elif metric == "euclidean":
            return self._euclidean_distance(embedding1, embedding2)
        elif metric == "dot":
            return -self._dot_product(
                embedding1, embedding2
            )  # Negative because smaller is better for distance
        else:
            raise ValueError(f"Unknown distance metric: {metric}")

    def calculate_similarity(self, text1: str, text2: str, metric: str = None) -> float:
        """
        Calculate the similarity between two texts.

        Args:
            text1: First text
            text2: Second text
            metric: Distance metric to use ('cosine', 'euclidean', or 'dot'). If None,
                   uses the metric specified during initialization.

        Returns:
            float: Similarity between the two texts (higher is more similar)

        Raises:
            ValueError: If the specified metric is not supported
        """
        # Get embeddings
        embedding1 = self.get_embedding(text1)
        embedding2 = self.get_embedding(text2)

        # Use provided metric or fall back to default
        metric = metric or self.distance_metric

        # Calculate similarity based on selected metric
        if metric == "cosine":
            return self._cosine_similarity(embedding1, embedding2)
        elif metric == "euclidean":
            # Convert euclidean distance to similarity (1 / (1 + distance))
            distance = self._euclidean_distance(embedding1, embedding2)
            return 1.0 / (1.0 + distance)
        elif metric == "dot":
            return self._dot_product(embedding1, embedding2)
        else:
            raise ValueError(f"Unknown distance metric: {metric}")

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two vectors.

        Args:
            vec1: First vector
            vec2: Second vector

        Returns:
            float: Cosine similarity (0-1)
        """
        # Ensure vectors are 1D
        vec1 = np.asarray(vec1).flatten()
        vec2 = np.asarray(vec2).flatten()

        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    def _euclidean_distance(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Calculate Euclidean distance between two vectors.

        Args:
            vec1: First vector
            vec2: Second vector

        Returns:
            float: Euclidean distance
        """
        # Ensure vectors are 1D
        vec1 = np.asarray(vec1).flatten()
        vec2 = np.asarray(vec2).flatten()

        return np.linalg.norm(vec1 - vec2)

    def _dot_product(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Calculate dot product between two vectors.

        Args:
            vec1: First vector
            vec2: Second vector

        Returns:
            float: Dot product
        """
        # Ensure vectors are 1D
        vec1 = np.asarray(vec1).flatten()
        vec2 = np.asarray(vec2).flatten()

        return np.dot(vec1, vec2)

    def save_to_file(self, filename: str) -> None:
        """
        Save embeddings to a JSON file.

        Args:
            filename: Path to save the embeddings to
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)

        # Convert numpy arrays to lists for JSON serialization
        serializable_embeddings = {}
        for key, embedding in self._embeddings.items():
            serializable_embeddings[key] = embedding.tolist()

        # Save embeddings and metadata
        data = {
            "embedding_provider": self.embedding_provider,
            "model_name": self.model_name,
            "distance_metric": self.distance_metric,
            "embeddings": serializable_embeddings,
        }

        with open(filename, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def load_from_file(self, filename: str) -> None:
        """
        Load embeddings from a JSON file.

        Args:
            filename: Path to load the embeddings from

        Raises:
            FileNotFoundError: If the file doesn't exist
            ValueError: If the file contains incompatible data
        """
        if not os.path.exists(filename):
            raise FileNotFoundError(f"Embeddings file not found: {filename}")

        try:
            with open(filename, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            raise ValueError(f"Error loading embeddings file: {str(e)}")

        file_provider = data.get("embedding_provider")
        file_model_name = data.get("model_name")

        if file_provider and file_provider != self.embedding_provider:
            print(
                f"Warning: Loading embeddings from a file generated with provider '{file_provider}', "
                f"but current Embedder is configured for '{self.embedding_provider}'. "
                f"Model compatibility and semantic consistency are not guaranteed."
            )

        if file_model_name and file_model_name != self.model_name:
            # More nuanced check considering provider might have changed expectations
            # If providers match, then model names should ideally match or be compatible.
            # If providers differ, this warning is secondary to the provider mismatch warning.
            warning_message = f"Model name mismatch: '{file_model_name}' (file) vs '{self.model_name}' (current). "
            if file_provider == self.embedding_provider:
                warning_message += "Embeddings may not be compatible."
            else:
                warning_message += "This might be expected if provider also changed. Ensure compatibility."
            print(f"Warning: {warning_message}")
            # Decided against raising ValueError for model mismatch if provider also mismatch or for general warning
            # The user is warned, and can decide if the loaded embeddings are usable.
            # However, if providers are the same and models differ, it's a stronger case for an error.
            if (
                file_provider == self.embedding_provider
                and file_model_name != self.model_name
            ):
                raise ValueError(
                    f"Model name mismatch for the same provider '{self.embedding_provider}': "
                    f"'{file_model_name}' (file) vs '{self.model_name}' (current). "
                    "These embeddings are likely incompatible."
                )

        # Convert list embeddings back to numpy arrays
        self._embeddings = {}
        for key, embedding_list in data.get("embeddings", {}).items():
            self._embeddings[key] = np.array(embedding_list)

        # Update distance metric if it exists in the file
        if "distance_metric" in data:
            self.distance_metric = data["distance_metric"]

    def clear_cache(self) -> None:
        """Clear the embedding cache."""
        self._embeddings = {}
