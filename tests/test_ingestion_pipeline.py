# from ard.data.dataset_item import DatasetItem
# from ard.data.metadata import Metadata
# from ard.knowledge_graph.knowledge_graph import KnowledgeGraph
# from ard.pipelines.ingestion_pipeline import run_ingestion_pipeline


# def test_ingestion_pipeline():
#     """Test that the ingestion pipeline runs and returns expected objects."""
#     # Run the pipeline
#     dataset_items, knowledge_graph = run_ingestion_pipeline()

#     # Check dataset items
#     assert len(dataset_items) == 3
#     assert all(isinstance(item, DatasetItem) for item in dataset_items)
#     assert {item.id for item in dataset_items} == {"doc1", "doc2", "doc3"}

#     # Check metadata
#     assert all(isinstance(item.get_metadata(), Metadata) for item in dataset_items)
#     assert dataset_items[0].get_metadata().doi == "10.1234/journal.prot.12345"
#     assert dataset_items[1].get_metadata().pm_id == "98765432"
#     assert dataset_items[2].get_metadata().doi == "10.5678/journal.cell.54321"
#     assert dataset_items[2].get_metadata().pm_id == "12345678"

#     # Check additional metadata
#     assert "journal" in dataset_items[0].get_metadata().additional_metadata
#     assert "keywords" in dataset_items[1].get_metadata().additional_metadata
#     assert dataset_items[2].get_metadata().additional_metadata["year"] == 2021

#     # Check knowledge graph
#     assert isinstance(knowledge_graph, KnowledgeGraph)
#     assert len(knowledge_graph.get_nodes()) == 3
#     assert len(knowledge_graph.get_edges()) == 2

#     # Check specific edges
#     edges = knowledge_graph.get_edges()
#     assert ("doc1", "doc2", "relates_to") in edges
#     assert ("doc2", "doc3", "cites") in edges
