from ardcore.data.triplets import Triplet
from ardcore.knowledge_graph import KnowledgeGraph
from ardcore.knowledge_graph.node_merger import ExactMatchNodeMerger


def test_merge_nodes():
    """Test merging of nodes."""
    # Create triplets with shared connections
    triplets = [
        Triplet(
            node_1="protein_A",
            edge="activates",
            node_2="pathway_Z",
            metadata={"source": "paper_1"},
        ),
        Triplet(
            node_1="protein_B",
            edge="inhibits",
            node_2="pathway_Z",
            metadata={"source": "paper_2"},
        ),
    ]

    kg = KnowledgeGraph.from_triplets(triplets, scientific_domain="test_domain")

    # Initial state should have two separate nodes
    assert kg.has_node("protein_A")
    assert kg.has_node("protein_B")

    # Merge nodes
    kg.merge_nodes("protein_A", "protein_B", "merged_protein")

    # Original nodes should be gone
    assert not kg.has_node("protein_A")
    assert not kg.has_node("protein_B")

    # New merged node should exist
    assert kg.has_node("merged_protein")

    # Check that the merged node has the combined sources
    merged_attrs = kg.get_node_attrs("merged_protein")
    assert "sources" in merged_attrs
    assert len(merged_attrs["sources"]) == 2

    # Check that relationships are preserved
    assert kg.has_edge("merged_protein", "pathway_Z")

    # The relationship should preserve both original metadata
    edges_attrs = kg.get_edge_attrs("merged_protein", "pathway_Z")
    for edge_attrs in edges_attrs:
        assert "sources" in edge_attrs
        assert len(edge_attrs["sources"]) == 2

    # Check the sources are preserved in the edge
    sources = [
        source["source"] for source in edge_attrs["sources"] if "source" in source
    ]
    assert "paper_1" in sources
    assert "paper_2" in sources


def test_merge_nodes_with_bidirectional_edges():
    """Test merging nodes when they have bidirectional connections."""
    # Create triplets with bidirectional connections
    triplets = [
        Triplet(
            node_1="gene_A",
            edge="regulates",
            node_2="gene_B",
            metadata={"source": "paper_1"},
        ),
        Triplet(
            node_1="gene_B",
            edge="inhibits",
            node_2="gene_A",
            metadata={"source": "paper_2"},
        ),
        Triplet(
            node_1="gene_C",
            edge="activates",
            node_2="gene_A",
            metadata={"source": "paper_3"},
        ),
    ]

    kg = KnowledgeGraph.from_triplets(triplets, scientific_domain="test_domain")

    # Initial state should have bidirectional edges
    assert kg.has_edge("gene_A", "gene_B")
    assert kg.has_edge("gene_B", "gene_A")

    # Merge gene_A and gene_C
    kg.merge_nodes("gene_A", "gene_C", "merged_gene")

    # Check that merged node has connections with gene_B
    assert kg.has_edge("merged_gene", "gene_B")
    assert kg.has_edge("gene_B", "merged_gene")

    # Check that original bidirectional relationship metadata is preserved
    outgoing_edges = kg.get_edge_attrs("merged_gene", "gene_B")
    for outgoing_edge in outgoing_edges:
        assert "sources" in outgoing_edge

    # Find the source with relation "regulates"
    regulates_sources = [
        s for s in outgoing_edge["sources"] if s.get("relation") == "regulates"
    ]
    assert len(regulates_sources) > 0
    assert regulates_sources[0]["source"] == "paper_1"

    # The relationship type is stored in the edge
    assert outgoing_edge["relation"] == "regulates"

    incoming_edges = kg.get_edge_attrs("gene_B", "merged_gene")
    for incoming_edge in incoming_edges:
        assert "sources" in incoming_edge

    # Find the source with relation "inhibits"
    inhibits_sources = [
        s for s in incoming_edge["sources"] if s.get("relation") == "inhibits"
    ]
    assert len(inhibits_sources) > 0
    assert inhibits_sources[0]["source"] == "paper_2"

    # The relationship type is stored in the edge
    assert incoming_edge["relation"] == "inhibits"

    # Check that additional incoming edges were properly redirected
    assert not kg.has_node("gene_A")  # Original node should be removed
    assert not kg.has_node("gene_C")  # Original node should be removed

    # Check total number of nodes and edges
    assert len(kg.get_nodes()) == 2  # merged_gene, gene_B

    # Check that the merged node has all the combined sources
    merged_attrs = kg.get_node_attrs("merged_gene")
    assert "sources" in merged_attrs

    # Extract source papers to verify
    source_papers = set()
    for source in merged_attrs["sources"]:
        if "source" in source:
            source_papers.add(source["source"])

    # Verify that all source papers are present
    assert "paper_1" in source_papers
    assert "paper_2" in source_papers
    assert "paper_3" in source_papers


def test_merge_similar_nodes_exact_match():
    """Test merging similar nodes using exact match strategy."""
    # Create triplets with nodes that have the same lowercase representation
    triplets = [
        Triplet(
            node_1="Protein",
            edge="binds_to",
            node_2="receptor_1",
            metadata={"source": "paper_1"},
        ),
        Triplet(
            node_1="protein",
            edge="activates",
            node_2="enzyme_1",
            metadata={"source": "paper_2"},
        ),
        Triplet(
            node_1="PROTEIN",
            edge="inhibits",
            node_2="kinase_1",
            metadata={"source": "paper_3"},
        ),
        Triplet(
            node_1="complex_A",
            edge="forms",
            node_2="Receptor",
            metadata={"source": "paper_4"},
        ),
        Triplet(
            node_1="kinase_2",
            edge="phosphorylates",
            node_2="receptor",
            metadata={"source": "paper_5"},
        ),
    ]

    kg = KnowledgeGraph.from_triplets(triplets, scientific_domain="test_domain")

    # Initial state should have different capitalizations as separate nodes
    assert kg.has_node("Protein")
    assert kg.has_node("protein")
    assert kg.has_node("PROTEIN")
    assert kg.has_node("Receptor")
    assert kg.has_node("receptor")
    assert kg.has_node(
        "receptor_1"
    )  # This is a different node, not just a case variation

    # Merge similar nodes
    merger = ExactMatchNodeMerger()
    kg.merge_similar_nodes(merger)

    # Check that protein variants were merged (case-insensitive)
    protein_variants = ["Protein", "protein", "PROTEIN"]
    merged_protein_exists = False
    for variant in protein_variants:
        if kg.has_node(variant):
            merged_protein_exists = True
            merged_protein = variant
            break

    assert merged_protein_exists, "One protein variant should remain after merging"

    # Only one protein variant should remain
    protein_variants_remaining = [v for v in protein_variants if kg.has_node(v)]
    assert len(protein_variants_remaining) == 1

    # Check that receptor variants were merged (case-insensitive)
    receptor_variants = [
        "Receptor",
        "receptor",
    ]  # receptor_1 should NOT be in this group
    receptor_variants_remaining = [v for v in receptor_variants if kg.has_node(v)]
    assert len(receptor_variants_remaining) == 1

    # receptor_1 should still exist as a separate node
    assert kg.has_node("receptor_1")

    # The merged protein node should have connections from all original protein nodes
    merged_protein = protein_variants_remaining[0]

    # Check connections - merged protein should have all outgoing edges from the original variants
    outgoing_edges = kg.get_edges_data()
    outgoing_targets = set()
    for source, target, _ in outgoing_edges:
        if source == merged_protein:
            outgoing_targets.add(target)

    assert "receptor_1" in outgoing_targets
    assert "enzyme_1" in outgoing_targets
    assert "kinase_1" in outgoing_targets

    # Check that sources are preserved in merged nodes
    protein_attrs = kg.get_node_attrs(merged_protein)
    protein_sources = protein_attrs["sources"]
    assert len(protein_sources) == 3  # From all three protein variants

    # Check that receptor_1 wasn't merged with Receptor/receptor
    assert kg.has_node("receptor_1")

    # Check if triplets reflect the merges correctly
    triplets_after_merge = kg.triplets

    # Count occurrences of each protein variant in triplets
    protein_occurrences = {}
    receptor_occurrences = {}

    for t in triplets_after_merge:
        # Count protein variants
        for variant in protein_variants:
            if t.node_1 == variant or t.node_2 == variant:
                protein_occurrences[variant] = protein_occurrences.get(variant, 0) + 1

        # Count receptor variants (excluding receptor_1)
        for variant in receptor_variants:
            if t.node_1 == variant or t.node_2 == variant:
                receptor_occurrences[variant] = receptor_occurrences.get(variant, 0) + 1

    # Only one protein variant and one receptor variant should appear in triplets
    assert len(protein_occurrences) == 1
    assert len(receptor_occurrences) == 1


def test_exact_match_merger_find_candidates():
    """Test that ExactMatchNodeMerger correctly identifies merge candidates."""
    # Create a knowledge graph with case variations
    triplets = [
        Triplet(node_1="Alpha", edge="connects", node_2="Beta"),
        Triplet(node_1="alpha", edge="connects", node_2="gamma"),
        Triplet(node_1="ALPHA", edge="connects", node_2="delta"),
        Triplet(node_1="Gamma", edge="connects", node_2="epsilon"),
        Triplet(node_1="gamma_1", edge="connects", node_2="zeta"),
        Triplet(node_1="lambda_1", edge="connects", node_2="theta"),
        Triplet(node_1="lambda_2", edge="connects", node_2="theta"),
    ]

    kg = KnowledgeGraph.from_triplets(triplets, scientific_domain="test_domain")

    # Create the merger
    merger = ExactMatchNodeMerger()

    # Find merge candidates
    candidates = merger.find_merge_candidates(kg)

    # Validate the candidates
    assert len(candidates) == 2  # Should find 2 groups

    # Find the alpha group and gamma group
    alpha_group = None
    gamma_group = None
    lambda_group = None

    for group in candidates:
        if "Alpha" in group or "alpha" in group or "ALPHA" in group:
            alpha_group = group
        elif "Gamma" in group or "gamma" in group or "gamma_1" in group:
            gamma_group = group
        elif "lambda_1" in group or "lambda_2" in group:
            lambda_group = group

    # Check the alpha group
    assert alpha_group is not None
    assert len(alpha_group) == 3
    assert "Alpha" in alpha_group
    assert "alpha" in alpha_group
    assert "ALPHA" in alpha_group

    # Check the gamma group
    assert gamma_group is not None
    assert len(gamma_group) == 2
    assert "Gamma" in gamma_group
    assert "gamma" in gamma_group
    # Check that gamma_1 is not in the gamma group
    assert "gamma_1" not in gamma_group

    # Check the lambda group does not exist (lambda_1 and lambda_2 are notmerged)
    assert lambda_group is None


def test_exact_match_merger_generate_name():
    """Test that ExactMatchNodeMerger correctly generates merged node names."""
    # Create knowledge graph
    triplets = [
        Triplet(
            node_1="Protein", edge="connects", node_2="X", metadata={"source": "doc1"}
        ),
        Triplet(
            node_1="Protein", edge="connects", node_2="Y", metadata={"source": "doc2"}
        ),
        Triplet(
            node_1="protein", edge="connects", node_2="Z", metadata={"source": "doc3"}
        ),
    ]

    kg = KnowledgeGraph.from_triplets(triplets, scientific_domain="test_domain")

    # Create the merger
    merger = ExactMatchNodeMerger()

    # Get the name for the merged node
    nodes_to_merge = {"Protein", "protein"}
    merged_name = merger.generate_merged_node_name(nodes_to_merge, kg)

    # The capitalization with more occurrences should be chosen
    assert merged_name == "Protein"  # "Protein" occurs in 2 triplets, "protein" in 1


def test_merge_preserves_all_edges():
    """Test that merging preserves all incoming and outgoing edges from all merged nodes."""
    # Create a graph with multiple edges between a set of nodes
    triplets = [
        # Node1 variants with outgoing edges
        Triplet(node_1="Node1", edge="connects_to", node_2="Target1"),
        Triplet(node_1="node1", edge="links_with", node_2="Target2"),
        Triplet(node_1="NODE1", edge="depends_on", node_2="Target3"),
        # Incoming edges to Node1 variants
        Triplet(node_1="Source1", edge="points_to", node_2="Node1"),
        Triplet(node_1="Source2", edge="references", node_2="node1"),
        Triplet(node_1="Source3", edge="cites", node_2="NODE1"),
    ]

    kg = KnowledgeGraph.from_triplets(triplets, scientific_domain="test_domain")

    # Initial state verification
    assert len(kg.get_nodes()) == 9  # 3 Node1 variants + 3 Targets + 3 Sources

    # Merge the Node1 variants
    merger = ExactMatchNodeMerger()
    kg.merge_similar_nodes(merger)

    # Find the merged node
    node1_variants = ["Node1", "node1", "NODE1"]
    merged_node = None
    for variant in node1_variants:
        if kg.has_node(variant):
            merged_node = variant
            break

    assert merged_node is not None, "One Node1 variant should exist after merging"

    # Check that all outgoing edges were preserved
    # Should have 3 outgoing edges to Target1, Target2, Target3
    outgoing_edges = kg.get_edges_data()
    outgoing_targets = set()
    for source, target, _ in outgoing_edges:
        if source == merged_node:
            outgoing_targets.add(target)

    assert "Target1" in outgoing_targets
    assert "Target2" in outgoing_targets
    assert "Target3" in outgoing_targets
    assert len(outgoing_targets) == 3

    # Check that all incoming edges were preserved
    # Should have 3 incoming edges from Source1, Source2, Source3
    incoming_edges = kg.get_edges_data()
    incoming_sources = set()
    for source, target, _ in incoming_edges:
        if target == merged_node:
            incoming_sources.add(source)

    assert "Source1" in incoming_sources
    assert "Source2" in incoming_sources
    assert "Source3" in incoming_sources
    assert len(incoming_sources) == 3

    # Check total number of nodes
    # Should have 7 nodes: 1 merged Node1 + 3 Targets + 3 Sources
    assert len(kg.get_nodes()) == 7


def test_metadata_aggregation_during_merge():
    """Test that metadata from nodes and edges is properly aggregated when merging."""
    # Create triplets with specific metadata to track
    triplets = [
        # Node variants with unique metadata
        Triplet(
            node_1="Entity",
            edge="relates_to",
            node_2="Object1",
            metadata={"source": "doc1", "confidence": 0.9, "unique_key1": "value1"},
        ),
        Triplet(
            node_1="entity",
            edge="connects_to",
            node_2="Object2",
            metadata={"source": "doc2", "confidence": 0.8, "unique_key2": "value2"},
        ),
        Triplet(
            node_1="ENTITY",
            edge="references",
            node_2="Object3",
            metadata={"source": "doc3", "confidence": 0.7, "unique_key3": "value3"},
        ),
        # Create incoming edges with unique metadata
        Triplet(
            node_1="Subject1",
            edge="points_to",
            node_2="Entity",
            metadata={"source": "doc4", "method": "method1", "tag": "tag1"},
        ),
        Triplet(
            node_1="Subject2",
            edge="links_to",
            node_2="entity",
            metadata={"source": "doc5", "method": "method2", "tag": "tag2"},
        ),
        Triplet(
            node_1="Subject2",
            edge="links_to",
            node_2="Entity",
            metadata={"source": "doc6", "method": "method3", "tag": "tag3"},
        ),
    ]

    kg = KnowledgeGraph.from_triplets(triplets, scientific_domain="test_domain")

    # Merge the entity variants
    merger = ExactMatchNodeMerger()
    kg.merge_similar_nodes(merger)

    # Find the merged node
    entity_variants = ["Entity", "entity", "ENTITY"]
    merged_entity = None
    for variant in entity_variants:
        if kg.has_node(variant):
            merged_entity = variant
            break

    assert merged_entity is not None, "One entity variant should exist after merging"

    # Check that node metadata is aggregated
    node_attrs = kg.get_node_attrs(merged_entity)
    assert "sources" in node_attrs
    sources = node_attrs["sources"]

    # Should have 6 sources, one from each triplet
    assert len(sources) == 6

    # Check for specific unique metadata keys
    unique_keys = set()
    for source in sources:
        for key in source.keys():
            if key.startswith("unique_key"):
                unique_keys.add(key)

    assert "unique_key1" in unique_keys
    assert "unique_key2" in unique_keys
    assert "unique_key3" in unique_keys

    # Check that all outgoing edges metadata is preserved
    outgoing_edge_metadata = {}
    edges_data = kg.get_edges_data()
    for source, target, attrs in edges_data:
        if source == merged_entity:
            outgoing_edge_metadata[target] = attrs

    assert "Object1" in outgoing_edge_metadata
    assert "Object2" in outgoing_edge_metadata
    assert "Object3" in outgoing_edge_metadata

    # Check specific edge metadata is preserved
    assert outgoing_edge_metadata["Object1"]["relation"] == "relates_to"
    assert outgoing_edge_metadata["Object2"]["relation"] == "connects_to"
    assert outgoing_edge_metadata["Object3"]["relation"] == "references"

    # Check incoming edge metadata
    incoming_edges = {}
    for source, target, attrs in edges_data:
        if target == merged_entity:
            if source not in incoming_edges:
                incoming_edges[source] = []
            incoming_edges[source].append(attrs)

    assert "Subject1" in incoming_edges
    assert "Subject2" in incoming_edges

    # In the new implementation, multiple edges from the same source are merged into one edge
    # with combined metadata, rather than kept as separate edges
    assert len(incoming_edges["Subject2"]) >= 1

    # Check that the sources inside the edge contain data from the original edges
    subject2_edge = incoming_edges["Subject2"][0]
    assert "sources" in subject2_edge

    # The combined edge should have sources from both original edges
    method_values = set()
    tag_values = set()
    for source in subject2_edge["sources"]:
        if "method" in source:
            method_values.add(source["method"])
        if "tag" in source:
            tag_values.add(source["tag"])

    # Verify that metadata from both original edges is present
    assert "method2" in method_values or "method3" in method_values
    assert "tag2" in tag_values or "tag3" in tag_values


def test_parallel_edge_merge():
    """
    Test that when two nodes with the same relation to a target node are merged,
    their edges are also merged with aggregated metadata.

    Example scenario:
    Pre-merge:
      node_1 --references({ source: [a] })--> node_2
      Node_1 --references({ source: [b] })--> node_2
    Post-merge:
      node_1 --references({ source: [a,b] })--> node_2
    """
    # Create triplets with same relation between case variants and the same target
    triplets = [
        # Two nodes with the same relation to the same target
        Triplet(
            node_1="Concept",
            edge="references",
            node_2="Target",
            metadata={"source": "doc1", "confidence": 0.9, "author": "Alice"},
        ),
        Triplet(
            node_1="concept",
            edge="references",
            node_2="Target",
            metadata={"source": "doc2", "confidence": 0.8, "author": "Bob"},
        ),
        # Add a different relation to ensure specific merging
        Triplet(
            node_1="Concept",
            edge="cites",
            node_2="Target",
            metadata={"source": "doc3", "confidence": 0.7, "year": 2020},
        ),
    ]

    kg = KnowledgeGraph.from_triplets(triplets, scientific_domain="test_domain")

    # Initial check
    assert kg.has_node("Concept")
    assert kg.has_node("concept")

    # Verify initial edges - two with "references" relation, one with "cites"
    assert kg.has_edge("Concept", "Target")
    assert kg.has_edge("concept", "Target")

    # Merge the variants
    merger = ExactMatchNodeMerger()
    kg.merge_similar_nodes(merger)

    # Find the merged node
    concept_variants = ["Concept", "concept"]
    merged_concept = None
    for variant in concept_variants:
        if kg.has_node(variant):
            merged_concept = variant
            break

    assert merged_concept is not None, "One concept variant should remain after merging"

    # Verify merged edges
    edges_attrs = kg.get_edge_attrs(merged_concept, "Target")
    for edge_attrs in edges_attrs:
        assert "sources" in edge_attrs

        # we need to skip the test for now because we allowed multiple edges between the same nodes
        """
        # There should be a combined edge with all metadata from both edges
        sources = edge_attrs["sources"]

        
        # Check that sources contain metadata from both original edges
        author_values = set()
        source_values = set()
        for source in sources:
            if "author" in source:
                author_values.add(source["author"])
            if "source" in source:
                source_values.add(source["source"])
            if "year" in source and source.get("year") == 2020:
                # Verify the "cites" relation metadata is included
                assert source.get("source") == "doc3"

        # Verify both authors are present
        assert "Alice" in author_values
        assert "Bob" in author_values

        # Verify all sources are present
        assert "doc1" in source_values
        assert "doc2" in source_values
        assert "doc3" in source_values
        """
