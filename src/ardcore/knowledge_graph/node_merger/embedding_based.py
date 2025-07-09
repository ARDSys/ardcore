import collections  # Added for defaultdict and deque
import os
from concurrent.futures import ThreadPoolExecutor, as_completed  # Added for parallelism
from typing import List, Literal, Optional, Set

import numpy as np
from loguru import logger
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

from ardcore.knowledge_graph.node_merger.base import NodeMerger
from ardcore.utils.embedder import Embedder


class EmbeddingBasedNodeMerger(NodeMerger):
    """
    Merges nodes based on embedding similarity.

    This merger uses a sentence transformer model to generate embeddings for node names
    and identifies nodes with similarity above a threshold for merging.
    It supports multiple strategies for finding merge candidates.
    """

    def __init__(
        self,
        embedding_provider: Literal["gemini", "sentence_transformer"] = "gemini",
        embedding_model_name: Optional[str] = None,
        similarity_threshold: float = 0.85,
        distance_metric: Literal["cosine", "euclidean", "dot"] = "cosine",
        embedder_path: Optional[str] = None,
        candidate_finder_strategy: Literal["brute_force", "ann"] = "brute_force",
        ann_k: int = 50,
    ):
        """
        Initialize the embedding-based node merger.

        Args:
            embedding_provider: The provider for generating embeddings.
            embedding_model_name: Name of the SentenceTransformer model or Gemini model.
            similarity_threshold: Threshold for similarity (0-1) to consider nodes for merging.
            distance_metric: Metric used by the embedder.
            embedder_path: Optional path to load/save embedder state.
            candidate_finder_strategy: Strategy to find merge candidates ('brute_force' or 'ann').
            ann_k: Number of nearest neighbors to retrieve if 'ann' strategy is used.
        """
        self.similarity_threshold = similarity_threshold
        self.embedder = Embedder(
            embedding_provider=embedding_provider,
            model_name=embedding_model_name,
            distance_metric=distance_metric,
        )
        self.embedder_path: Optional[str] = None
        if embedder_path:
            self.embedder_path = embedder_path
            if os.path.exists(embedder_path):
                self.embedder.load_from_file(embedder_path)
            else:
                logger.warning(
                    f"Embedder path {embedder_path} does not exist, will compute embeddings on the fly"
                )

        self.candidate_finder_strategy = candidate_finder_strategy
        if not isinstance(ann_k, int) or ann_k <= 0:
            logger.warning(
                f"ann_k must be a positive integer. Got {ann_k}. Defaulting to 50."
            )
            self.ann_k = 50
        else:
            self.ann_k = ann_k

        self.distance_metric = distance_metric  # Store for ANN metric selection

    def find_merge_candidates(self, knowledge_graph) -> List[Set[str]]:
        """
        Find groups of nodes with similar embeddings using the configured strategy.

        Args:
            knowledge_graph: The KnowledgeGraph instance

        Returns:
            List[Set[str]]: List of sets where each set contains node names to be merged
        """
        nodes = list(knowledge_graph.get_nodes())
        if not nodes:
            logger.info("No nodes in the graph to process for merging.")
            return []

        logger.info(
            f"Processing {len(nodes)} nodes for merge candidates using '{self.candidate_finder_strategy}' strategy."
        )
        # Ensure embeddings are computed/loaded for all nodes once.
        # The Embedder class should handle caching internally.
        _ = self.embedder.get_embeddings(nodes, show_progress_bar=True)
        logger.info(f"Embeddings are ready for {len(nodes)} nodes.")

        if self.embedder_path:
            self.embedder.save_to_file(self.embedder_path)
            logger.info(f"Embedder state saved to {self.embedder_path}")

        if self.candidate_finder_strategy == "brute_force":
            return self._find_merge_candidates_brute_force(knowledge_graph, nodes)
        elif self.candidate_finder_strategy == "ann":
            return self._find_merge_candidates_ann(knowledge_graph, nodes)
        else:
            raise ValueError(
                f"Unknown candidate_finder_strategy: {self.candidate_finder_strategy}"
            )

    def _find_merge_candidates_brute_force(
        self, knowledge_graph, nodes: List[str]
    ) -> List[Set[str]]:
        """
        Finds merge candidates using a brute-force O(N^2) comparison.
        """
        similar_groups = []
        processed_nodes = set()

        for i in tqdm(range(len(nodes)), desc="Finding merge candidates (Brute Force)"):
            node1 = nodes[i]
            if node1 in processed_nodes:
                continue

            current_group = {node1}
            for j in range(i + 1, len(nodes)):
                node2 = nodes[j]
                if node2 in processed_nodes:  # Already part of another group
                    continue

                similarity = self.embedder.calculate_similarity(node1, node2)
                if similarity >= self.similarity_threshold:
                    current_group.add(node2)

            if len(current_group) > 1:
                similar_groups.append(current_group)
                processed_nodes.update(current_group)
            else:  # node1 did not form a group with any other unprocessed nodes
                processed_nodes.add(node1)

        logger.info(
            f"Brute-force strategy found {len(similar_groups)} merge candidates."
        )
        return similar_groups

    def _find_merge_candidates_ann(
        self, knowledge_graph, nodes: List[str]
    ) -> List[Set[str]]:
        """
        Finds merge candidates using Approximate Nearest Neighbors.
        This version parallelizes the precise similarity calculation step.
        """
        if not nodes:  # Early exit if no nodes
            logger.info("ANN: No nodes to process.")
            return []

        # 1. Get embeddings for all nodes (already cached by Embedder)
        embedding_dict = self.embedder.get_embeddings(
            nodes, show_progress_bar=False
        )  # Progress bar was True, changing to False as get_embeddings is called before too.

        if not embedding_dict or len(embedding_dict) < len(nodes):
            logger.error(
                f"ANN: Not all {len(nodes)} nodes have embeddings. "
                f"{len(embedding_dict)} embeddings received. Aborting ANN strategy."
            )
            return []

        # 2. Create the embedding matrix in the exact order of 'nodes'
        ordered_embedding_vectors = []
        for node_name in tqdm(
            nodes, desc="ANN: Creating embedding matrix", leave=False
        ):
            embedding = embedding_dict.get(node_name)
            if embedding is None:  # Should have been caught by the check above
                logger.error(
                    f"ANN: Critical - Missing embedding for node '{node_name}' when creating matrix. Aborting."
                )
                return []
            ordered_embedding_vectors.append(embedding)

        if not ordered_embedding_vectors:
            logger.info(
                "ANN: No embedding vectors to process after attempting to create matrix."
            )
            return []

        embedding_matrix = np.array(ordered_embedding_vectors)

        if embedding_matrix.ndim == 1 and len(nodes) > 0:
            if len(nodes) == 1:
                embedding_matrix = embedding_matrix.reshape(1, -1)
            else:
                logger.error(
                    f"ANN: Embedding matrix has unexpected shape {embedding_matrix.shape} for {len(nodes)} nodes. Aborting."
                )
                return []

        if (
            embedding_matrix.shape[0] < 1
        ):  # Changed from < 2, as 1 node is not an error, just no pairs.
            logger.info("ANN: Less than 1 node in matrix, no pairs to find.")
            return []
        if embedding_matrix.shape[0] == 1:
            logger.info("ANN: Only 1 node in matrix, no pairs to find.")
            return []

        # 3. Setup and fit NearestNeighbors model
        nn_metric_map = {"cosine": "cosine", "euclidean": "euclidean"}
        selected_nn_metric = nn_metric_map.get(self.distance_metric)
        if selected_nn_metric is None:
            logger.warning(
                f"ANN: Embedder's metric '{self.distance_metric}' not directly mapped for NearestNeighbors. "
                f"Defaulting to 'cosine'. Ensure embeddings are suitable."
            )
            selected_nn_metric = "cosine"

        num_query_neighbors = min(self.ann_k + 1, embedding_matrix.shape[0])
        if num_query_neighbors <= 1 and embedding_matrix.shape[0] > 1:
            logger.info(
                f"ANN: Adjusted k to find at least one other neighbor. Original ann_k: {self.ann_k}, "
                f"num_query_neighbors will be min(2, num_samples)."
            )
            num_query_neighbors = min(2, embedding_matrix.shape[0])

        if (
            num_query_neighbors == 0 and embedding_matrix.shape[0] > 0
        ):  # If no neighbors can be queried.
            logger.info("ANN: num_query_neighbors is 0. No pairs will be found.")
            return []

        nn_model = NearestNeighbors(
            n_neighbors=num_query_neighbors, metric=selected_nn_metric, n_jobs=-1
        )
        try:
            logger.info(
                f"ANN: Fitting NearestNeighbors model with k={num_query_neighbors} for {embedding_matrix.shape[0]} items."
            )
            nn_model.fit(embedding_matrix)
        except ValueError as e:
            logger.error(
                f"ANN: Error fitting NearestNeighbors model: {e}. Check k or data."
            )
            return []

        # 4. Get all k-neighbors for all points (batched query)
        logger.info("ANN: Querying all k-neighbors...")
        try:
            # This single call should be efficient and potentially parallelized by scikit-learn
            all_distances, all_indices = nn_model.kneighbors(embedding_matrix)
        except ValueError as e:
            logger.error(f"ANN: Error in global kneighbors query: {e}. Aborting.")
            return []
        logger.info("ANN: k-neighbor querying complete.")

        # 5. Identify unique candidate pairs for precise similarity checking
        candidate_pairs_for_similarity_check = set()
        for i in range(len(nodes)):
            node1_name = nodes[i]
            for neighbor_original_idx in all_indices[i]:
                if i == neighbor_original_idx:  # Skip self
                    continue
                node2_name = nodes[neighbor_original_idx]
                # Add as a sorted tuple to ensure (A,B) is same as (B,A) and checked once
                pair = tuple(sorted((node1_name, node2_name)))
                candidate_pairs_for_similarity_check.add(pair)

        logger.info(
            f"ANN: Generated {len(candidate_pairs_for_similarity_check)} unique candidate pairs for precise check."
        )

        if not candidate_pairs_for_similarity_check:
            logger.info("ANN: No candidate pairs found from k-NN.")
            return []

        # 6. Parallelize precise similarity calculation for these candidate pairs
        actual_similar_pairs = []
        # Use ThreadPoolExecutor as calculate_similarity (with cached embeddings) is mostly numpy ops (GIL-releasing)
        # and avoids pickling overhead of ProcessPoolExecutor for the embedder object.
        num_workers = os.cpu_count()
        logger.info(
            f"ANN: Calculating precise similarities for {len(candidate_pairs_for_similarity_check)} pairs using up to {num_workers} workers."
        )

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {}
            for node1_name, node2_name in tqdm(
                list(candidate_pairs_for_similarity_check),
                desc="ANN: Precise similarity checks",
                leave=False,
            ):
                future = executor.submit(
                    self.embedder.calculate_similarity, node1_name, node2_name
                )
                futures[future] = (node1_name, node2_name)

            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="ANN: Collecting similarity results",
                leave=False,
            ):
                n1, n2 = futures[future]
                try:
                    similarity = future.result()
                    if similarity >= self.similarity_threshold:
                        actual_similar_pairs.append((n1, n2))
                except Exception as e:
                    logger.error(
                        f"ANN: Error calculating similarity for ({n1}, {n2}): {e}"
                    )

        logger.info(
            f"ANN: Found {len(actual_similar_pairs)} pairs meeting similarity threshold {self.similarity_threshold}."
        )
        if not actual_similar_pairs:
            return []

        # 7. Group nodes based on actual_similar_pairs (Connected Components)
        logger.info("ANN: Building merge groups from similar pairs...")
        adj = collections.defaultdict(set)
        all_involved_nodes_in_similarity_graph = set()
        for u, v in actual_similar_pairs:
            adj[u].add(v)
            adj[v].add(u)
            all_involved_nodes_in_similarity_graph.add(u)
            all_involved_nodes_in_similarity_graph.add(v)

        visited_nodes_for_grouping = set()
        final_groups = []

        # Iterate through original 'nodes' list to ensure deterministic group discovery order if node names had some implicit order
        # or just iterate through all_involved_nodes_in_similarity_graph for efficiency if order doesn't matter.
        # Using all_involved_nodes_in_similarity_graph is likely more direct for nodes known to be in similarity graph.

        # Processing nodes in the order they appear in the input `nodes` list can provide more deterministic output
        # if multiple grouping solutions are possible, though for connected components it usually doesn't change the groups themselves.
        # Using `nodes` ensures we consider every original node as a potential start for a group if it's involved.

        # Corrected iteration logic: Iterate through all nodes to correctly identify components.
        # Nodes not in all_involved_nodes_in_similarity_graph won't form groups of >1.
        for node_to_start_group_from in tqdm(
            nodes, desc="ANN: Forming groups (CCA)", leave=False
        ):
            if (
                node_to_start_group_from in all_involved_nodes_in_similarity_graph
                and node_to_start_group_from not in visited_nodes_for_grouping
            ):
                current_group = set()
                q = collections.deque()

                q.append(node_to_start_group_from)
                visited_nodes_for_grouping.add(node_to_start_group_from)
                current_group.add(node_to_start_group_from)

                while q:
                    curr_node_in_group = q.popleft()
                    for neighbor in adj[curr_node_in_group]:
                        if neighbor not in visited_nodes_for_grouping:
                            visited_nodes_for_grouping.add(neighbor)
                            current_group.add(neighbor)
                            q.append(neighbor)

                if len(current_group) > 1:  # Only add actual groups
                    final_groups.append(current_group)
            elif node_to_start_group_from not in visited_nodes_for_grouping:
                # This node was not involved in any similar pairs or already processed. Mark visited.
                visited_nodes_for_grouping.add(node_to_start_group_from)

        logger.info(
            f"ANN strategy found {len(final_groups)} merge candidates with k={self.ann_k} using parallelized similarity calculation."
        )
        return final_groups

    def generate_merged_node_name(self, nodes: Set[str], knowledge_graph) -> str:
        """
        Generate a name for the merged node, using the most frequent name.

        Args:
            nodes: Set of node names to be merged
            knowledge_graph: The KnowledgeGraph instance

        Returns:
            str: The name for the merged node
        """
        # Use the most frequently used node name
        node_counts = {}
        for node in nodes:
            node_attrs = knowledge_graph.get_node_attrs(node)
            count = len(node_attrs.get("sources", []))
            node_counts[node] = count

        # Return the node with highest count
        return max(node_counts.items(), key=lambda x: x[1])[0]
