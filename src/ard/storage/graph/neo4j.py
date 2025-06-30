import json
from typing import Any, Dict, List, Set, Tuple

from loguru import logger
from neo4j import GraphDatabase

from ard.storage.graph import GraphBackend


class Neo4jBackend(GraphBackend):
    """
    Neo4j backend implementation for the knowledge graph.
    """

    def __init__(
        self,
        uri: str = "bolt://localhost:7687",
        user: str = "neo4j",
        password: str = "password",
        database: str = "neo4j",
    ):
        """
        Initialize a new Neo4j backend.

        Args:
            uri (str): The Neo4j database URI
            user (str): The database user
            password (str): The database password
            database (str): The database name
        """
        self._driver = GraphDatabase.driver(uri, auth=(user, password))
        self._database = database
        self._ensure_constraints()

    @classmethod
    def from_serializable(cls, data: Dict[str, Any], **connection_params):
        """
        Initialize a new Neo4j backend from a serialized dictionary.

        This method loads data for all scientific domains present in the serialized data.
        Each node and edge will be created with its original scientific domain preserved
        in the attributes.

        Args:
            data (Dict[str, Any]): Serialized graph data from to_serializable()
            **connection_params: Neo4j connection parameters (uri, user, password, database)

        Returns:
            Neo4jBackend: New backend instance with all data loaded
        """
        # Create Neo4j backend with connection parameters
        backend = cls(**connection_params)

        # Add all nodes with their attributes
        for node_data in data["nodes"]:
            node_id = node_data["id"]
            attrs = node_data.get("attributes", {})
            backend.add_node(node_id, **attrs)

        # Add all edges with their attributes
        for edge_data in data["edges"]:
            source = edge_data["source"]
            target = edge_data["target"]
            attrs = edge_data.get("attributes", {})
            backend.add_edge(source, target, **attrs)

        return backend

    def to_serializable(self) -> Dict[str, Any]:
        """
        Convert the Neo4j graph to a serializable dictionary.

        This method exports all nodes and edges across all scientific domains present
        in the graph. The scientific domain information is preserved in the node labels
        and edge attributes of the exported data.

        Returns:
            Dict[str, Any]: A serializable representation of the graph containing all scientific domains
        """
        # Get all nodes with their attributes
        nodes = []
        for node in self.get_nodes():
            attrs = self.get_node_attrs(node)
            nodes.append({"id": node, "attributes": attrs})

        # Get all edges with their attributes
        edges = []
        for source, target, attrs in self.get_edges():
            edges.append({"source": source, "target": target, "attributes": attrs})

        return {"nodes": nodes, "edges": edges}

    def _ensure_constraints(self) -> None:
        """
        Ensure required constraints exist in the database.

        """
        pass

    def _close(self) -> None:
        """Close the Neo4j driver connection."""
        self._driver.close()

    def has_node(self, node_id: str, scientific_domain: str) -> bool:
        """
        Check if a node exists in the graph.

        Args:
            node_id: The name of the node
            scientific_domain: The scientific domain of the node

        Returns:
            bool: True if the node exists, False otherwise
        """
        query = f"MATCH (n:{scientific_domain}) WHERE n.name = $name RETURN count(n) AS count"

        with self._driver.session(database=self._database) as session:
            result = session.run(query, name=node_id)
            return result.single()["count"] > 0

    def add_node(self, node: str, scientific_domain: str, **attrs) -> None:
        """
        Create a node if it doesn't exist, or update an existing node.

        This function will:
        - Create a new node with the given scientific_domain and attributes if no node with this name exists
        - Add the specified scientific_domain to an existing node if it doesn't already have it
        - Update node properties in both cases

        Args:
            node: The name/identifier of the node
            scientific_domain: The scientific_domain to ensure the node has
            **attrs: Optional attributes to set on the node
        """
        with self._driver.session(database=self._database) as session:
            props = {k: v for k, v in attrs.items() if v is not None}
            props["name"] = node

            # Serialize complex data structures to JSON
            for key, value in props.items():
                if isinstance(value, (list, dict)) and key != "name":
                    props[key] = json.dumps(value)

            result = session.run(
                f"""
                MERGE (n {{name: $node}})
                ON CREATE SET n:{scientific_domain}, n += $props
                ON MATCH SET n:{scientific_domain}, n += $props
                RETURN n.name AS name, labels(n) AS labels
                """,
                node=node,
                props=props,
            )

            record = result.single()

            if record:
                node_name = record["name"]
                node_labels = record["labels"]
                logger.debug(
                    f"Node '{node_name}' created or updated with labels: {node_labels}"
                )
            else:
                logger.warning(f"Failed to create or update node '{node}'")

    def add_edge(
        self, source: str, target: str, relation: str, scientific_domain: str, **attrs
    ) -> None:
        """
        Create an edge between source and target nodes or update an existing edge.

        This function will:
        - Create a new edge with the given scientific_domain and attributes if no edge exists between the nodes
        - Update the edge properties if an edge already exists
        - Ensure both source and target nodes exist before creating the edge

        Args:
            source: The name of the source node
            target: The name of the target node
            relation: The relationship type
            scientific_domain: The scientific domain for the edge
            **attrs: Optional attributes to set on the edge
        """
        with self._driver.session(database=self._database) as session:
            # Convert attributes to Neo4j format, filtering out None values
            props = {k: v for k, v in attrs.items() if v is not None}
            props["edge"] = relation

            # Serialize complex data structures to JSON
            for key, value in props.items():
                if isinstance(value, (list, dict)) and key != "scientific_domain":
                    props[key] = json.dumps(value)

            # If scientific_domain is not in props, add it
            if "scientific_domain" not in props:
                props["scientific_domain"] = [scientific_domain]
            elif scientific_domain not in props["scientific_domain"]:
                # If scientific_domain not in existing scientific_domain list, add it
                props["scientific_domain"] = props["scientific_domain"] + [
                    scientific_domain
                ]

            # MERGE finds the relationship or creates it if it doesn't exist
            query = f"""
            MATCH (source), (target)
            WHERE source.name = "{source}" AND target.name = "{target}"
            MERGE (source)-[r:`{relation}`]->(target)
            SET r += $props
            RETURN type(r) as type, properties(r) as props
            """

            result = session.run(query, source=source, target=target, props=props)
            record = result.single()

            if record:
                logger.debug(
                    f"Edge from {source} to {target} created or updated with properties: {record['props']}"
                )
            else:
                logger.warning(
                    f"Failed to create or update edge from {source} to {target}"
                )

    def has_edge(self, source: str, target: str, scientific_domain: str) -> bool:
        """Check if an edge exists using name."""
        with self._driver.session(database=self._database) as session:
            result = session.run(
                f"""
                MATCH (source)-[r]->(target)
                WHERE source.name = $source AND target.name = $target AND '{scientific_domain}' IN r.scientific_domain
                RETURN count(r) as count
                """,
                source=source,
                target=target,
            )
            return result.single()["count"] > 0

    def get_node_attrs(self, node: str, scientific_domain: str) -> Dict[str, Any]:
        """Get all attributes of a node by its name."""
        with self._driver.session(database=self._database) as session:
            result = session.run(
                f"""
                MATCH (n:{scientific_domain})
                WHERE n.name = $id
                RETURN properties(n) as props
                """,
                id=node,
            )
            record = result.single()
            if not record:
                return {}

            props = record["props"].copy()

            # Deserialize JSON strings back to Python objects
            for key, value in props.items():
                if isinstance(value, str) and key != "name":
                    try:
                        # Try to parse as JSON
                        props[key] = json.loads(value)
                    except (json.JSONDecodeError, ValueError):
                        # If it's not valid JSON, keep as string
                        pass

            # If there's no 'sources' list but there are other properties that should be
            # in source metadata, create a sources list
            if (
                "sources" not in props and len(props) > 1
            ):  # More than just the name property
                # Create a normalized structure with sources
                normalized_props = {}

                # Move metadata to sources list
                source_entry = {
                    k: v for k, v in props.items() if k != "name" and k != "sources"
                }
                if source_entry:  # Only add sources if there's actual metadata
                    normalized_props["sources"] = [source_entry]

                return normalized_props

            return props

    def get_edge_attrs(
        self, source: str, target: str, scientific_domain: str
    ) -> List[Dict[str, Any]]:
        """
        Get attributes of all edges between source and target in the specified scientific domain.

        Args:
            source: The source node identifier
            target: The target node identifier
            scientific_domain: The scientific domain to filter by

        Returns:
            List[Dict[str, Any]]: List of attribute dictionaries for all matching edges
        """
        with self._driver.session(database=self._database) as session:
            result = session.run(
                f"""
                MATCH (source)-[r]->(target)
                WHERE source.name = $source AND target.name = $target AND '{scientific_domain}' IN r.scientific_domain
                RETURN properties(r) as props
                """,
                source=source,
                target=target,
            )

            edges_attrs = []
            for record in result:
                props = record["props"]

                # Deserialize JSON strings back to Python objects
                for key, value in props.items():
                    if isinstance(value, str) and key not in ["edge", "relation"]:
                        try:
                            # Try to parse as JSON
                            props[key] = json.loads(value)
                        except (json.JSONDecodeError, ValueError):
                            # If it's not valid JSON, keep as string
                            pass

                # Normalize format to match NetworkX backend
                # If 'edge' exists but 'relation' doesn't, rename it
                if "edge" in props and "relation" not in props:
                    props["relation"] = props["edge"]

                # Create a sources array if it doesn't exist
                if "sources" not in props:
                    # Gather metadata from the edge itself to create a source entry
                    source_entry = {
                        k: v for k, v in props.items() if k not in ["relation"]
                    }
                    # Include the relation in the source entry
                    if "relation" in props:
                        source_entry["relation"] = props["relation"]
                    # Remove metadata keys from the top level, leaving only 'relation' and 'sources'
                    for k in list(props.keys()):
                        if k != "relation" and k != "sources":
                            props.pop(k)
                    # Add the source entry to the sources array
                    props["sources"] = [source_entry]

                edges_attrs.append(props)

            return edges_attrs

    def get_nodes(self, scientific_domain: str) -> Set[str]:
        """Get all nodes in the graph using name."""
        with self._driver.session(database=self._database) as session:
            result = session.run(f"MATCH (n:{scientific_domain}) RETURN n.name AS name")
            return {record["name"] for record in result}

    def get_random_node(self, scientific_domain: str) -> str:
        """Get a random node from the graph using name."""
        with self._driver.session(database=self._database) as session:
            result = session.run(
                f"""
                MATCH (n:{scientific_domain}) 
                RETURN n.name AS name
                ORDER BY rand()
                LIMIT 1
                """
            )
            record = result.single()
            return record["name"] if record else None

    def get_edges(
        self, scientific_domain: str
    ) -> List[Tuple[str, str, Dict[str, Any]]]:
        """Get all edges in the graph with their attributes using name."""
        with self._driver.session(database=self._database) as session:
            result = session.run(
                f"""
                MATCH (source)-[r]->(target)
                WHERE '{scientific_domain}' IN r.scientific_domain
                RETURN source.name as source, target.name as target, properties(r) AS props
                """
            )
            edges = []
            for record in result:
                source = record["source"]
                target = record["target"]
                props = record[
                    "props"
                ].copy()  # Make a copy to avoid modifying the original

                # Normalize format to match NetworkX backend
                normalized_props = {}

                # Handle relation/edge property
                relation_value = None
                if "edge" in props:
                    relation_value = props.pop("edge")
                    normalized_props["relation"] = relation_value
                elif "relation" in props:
                    relation_value = props["relation"]
                    normalized_props["relation"] = props.pop("relation")

                # Create a sources list if it doesn't exist
                if "sources" not in props:
                    # Use remaining properties as a single source entry
                    source_entry = props.copy()

                    # Add the relation to the source entry
                    if relation_value is not None:
                        source_entry["relation"] = relation_value
                        # Also add as 'edge' to match NetworkX format
                        source_entry["edge"] = relation_value

                    normalized_props["sources"] = [source_entry]
                else:
                    # If sources already exists, use it and ensure each entry has both relation and edge
                    sources = props.pop("sources")
                    for source_entry in sources:
                        if "relation" in source_entry and "edge" not in source_entry:
                            source_entry["edge"] = source_entry["relation"]
                        elif "edge" in source_entry and "relation" not in source_entry:
                            source_entry["relation"] = source_entry["edge"]

                    normalized_props["sources"] = sources

                    # Add any remaining top-level properties to normalized_props
                    for k, v in props.items():
                        normalized_props[k] = v

                edges.append((source, target, normalized_props))

            return edges

    def get_successors(
        self, node: str, scientific_domain: str, n_of_nodes: int = None
    ) -> List[str]:
        """Get all successor nodes of a node using name."""
        if n_of_nodes is None:
            query_limit = ""
        else:
            query_limit = f" LIMIT {n_of_nodes} "
        with self._driver.session(database=self._database) as session:
            result = session.run(
                f"""
                MATCH (source:{scientific_domain})-[]->(target:{scientific_domain})
                WHERE source.name = $id
                ORDER BY rand()
                {query_limit}
                RETURN target.name as id
                """,
                id=node,
            )
            # remove duplicates caused by multiple edges between the same nodes
            return list(set([record["id"] for record in result]))

    def get_predecessors(
        self, node: str, scientific_domain: str, n_of_nodes: int = None
    ) -> List[str]:
        """Get all predecessor nodes of a node using name."""
        if n_of_nodes is None:
            query_limit = ""
        else:
            query_limit = f" LIMIT {n_of_nodes} "

        with self._driver.session(database=self._database) as session:
            result = session.run(
                f"""
                MATCH (source:{scientific_domain})-[]->(target:{scientific_domain})
                WHERE target.name = $id
                ORDER BY rand()
                {query_limit}
                RETURN source.name as id
                """,
                id=node,
            )
            # remove duplicates caused by multiple edges between the same nodes
            return list(set([record["id"] for record in result]))

    def _normalize_edge_data(self, edges_data):
        """Normalize edge data to match NetworkX format, without adding triplet_ids."""
        normalized_edges = []

        for source, target, props in edges_data:
            props = props.copy()  # Work with a copy

            # Create a normalized structure
            normalized = {
                "relation": props.get("relation") or props.get("edge", ""),
                "sources": [],
            }

            # Process the sources list if it exists
            if "sources" in props:
                sources = props.pop("sources")
                for entry in sources:
                    # Ensure both relation and edge exist
                    if "relation" in entry and "edge" not in entry:
                        entry["edge"] = entry["relation"]
                    elif "edge" in entry and "relation" not in entry:
                        entry["relation"] = entry["edge"]

                    normalized["sources"].append(entry)
            else:
                # Create a source entry from the top-level properties
                source_entry = {k: v for k, v in props.items() if k not in ["relation"]}
                source_entry["relation"] = normalized["relation"]
                source_entry["edge"] = normalized["relation"]
                normalized["sources"].append(source_entry)

            normalized_edges.append((source, target, normalized))

        return normalized_edges

    def get_out_edges(
        self, node: str, scientific_domain: str
    ) -> List[Tuple[str, str, Dict[str, Any]]]:
        """Get all outgoing edges of a node with their attributes using name."""
        with self._driver.session(database=self._database) as session:
            query = f"""MATCH (source)-[r]->(target) WHERE source.name = '{node}' AND '{scientific_domain}' IN r.scientific_domain 
                RETURN source.name as source, target.name as target, properties(r) as props"""

            result = session.run(
                query,
                id=node,
            )
            edges = [(r["source"], r["target"], r["props"]) for r in result]
            return self._normalize_edge_data(edges)

    def get_in_edges(
        self, node: str, scientific_domain: str
    ) -> List[Tuple[str, str, Dict[str, Any]]]:
        """Get all incoming edges of a node with their attributes using name."""
        with self._driver.session(database=self._database) as session:
            query = f"""MATCH (source)-[r]->(target) WHERE target.name = '{node}' AND '{scientific_domain}' IN r.scientific_domain
                RETURN source.name as source, target.name as target, properties(r) as props"""

            result = session.run(
                query,
                id=node,
            )
            edges = [(r["source"], r["target"], r["props"].copy()) for r in result]
            return self._normalize_edge_data(edges)

    def remove_node(self, node: str, scientific_domain: str) -> None:
        """
        Remove a node from the specified scientific domain.

        This function will:
        - Remove the node entirely if it only belongs to the specified scientific domain
        - Otherwise, just remove the specified scientific domain from the node's labels

        Args:
            node: The node identifier
            scientific_domain: The scientific domain to remove the node from
        """
        with self._driver.session(database=self._database) as session:
            # First check how many labels the node has
            result = session.run(
                """
                MATCH (n)
                WHERE n.name = $id
                RETURN labels(n) as labels
                """,
                id=node,
            )
            record = result.single()

            if not record:
                # Node doesn't exist
                return

            labels = record["labels"]

            # Check if the node has the specified scientific_domain label
            if scientific_domain not in labels:
                # Node doesn't have this domain, nothing to do
                return

            # Count scientific domain labels (excluding built-in Neo4j labels)
            domain_labels = [
                label
                for label in labels
                if label != "Node" and not label.startswith("_")
            ]

            if len(domain_labels) <= 1:
                # If this is the only domain label, remove the node entirely
                session.run(
                    """
                    MATCH (n)
                    WHERE n.name = $id
                    DETACH DELETE n
                    """,
                    id=node,
                )
            else:
                # Just remove the specified scientific_domain label
                session.run(
                    f"""
                    MATCH (n)
                    WHERE n.name = $id
                    REMOVE n:{scientific_domain}
                    """,
                    id=node,
                )

    def number_of_edges(self, scientific_domain: str) -> int:
        """Get the total number of edges in the graph."""
        with self._driver.session(database=self._database) as session:
            result = session.run(
                f"MATCH ()-[r]->() WHERE '{scientific_domain}' IN r.scientific_domain RETURN count(r) AS count"
            )
            return result.single()["count"]

    def shortest_path(
        self, source: str, target: str, scientific_domain: str, directed: bool = True
    ) -> List[str]:
        """
        Get the shortest path between two nodes where edges belong to the specified scientific domain.

        Args:
            source: The name of the source node
            target: The name of the target node
            scientific_domain: The scientific domain to filter relationships
            directed: Whether to consider direction of edges (True) or treat them as undirected (False)
        """
        with self._driver.session(database=self._database) as session:
            if directed:
                query = f"""
                MATCH (source:{scientific_domain}), (target:{scientific_domain})
                WHERE source.name = $source AND target.name = $target
                MATCH p = shortestPath((source)-[*]->(target))
                WHERE ALL(r IN relationships(p) WHERE '{scientific_domain}' IN r.scientific_domain)
                RETURN [n IN nodes(p) | n.name] AS path
                """
            else:
                query = f"""
                MATCH (source:{scientific_domain}), (target:{scientific_domain})
                WHERE source.name = $source AND target.name = $target
                MATCH p = shortestPath((source)-[*]-(target))
                WHERE ALL(r IN relationships(p) WHERE '{scientific_domain}' IN r.scientific_domain)
                RETURN [n IN nodes(p) | n.name] AS path
                """
            print(query)
            result = session.run(query, source=source, target=target)
            record = result.single()
            return record["path"] if record else []

    def prepare_plain_bkps(self):
        """Get all relations in the graph. The function ise used for all scientific domains."""
        with self._driver.session(database=self._database) as session:
            pre_json_query = """MATCH (n)-[r]->(m)
                        RETURN 
                            n.name as source_name,
                            labels(n) as source_labels,
                            properties(n) as source_props,
                            type(r) as rel_type,
                            properties(r) as rel_props,
                            m.name as target_name,
                            labels(m) as target_labels,
                            properties(m) as target_props
                            """
            pre_json_result = list(session.run(pre_json_query))

            cypher_query = (
                """CALL apoc.export.cypher.all(null, {format: 'plain', stream: true})"""
            )
            cypher_result = session.run(cypher_query)

            all_statements = []
            for record in cypher_result:
                all_statements.append(record["cypherStatements"])

            cypher_script = "\n".join(all_statements)

        return pre_json_result, cypher_script

    def __len__(self, scientific_domain: str) -> int:
        """Get the total number of nodes in the graph."""
        with self._driver.session(database=self._database) as session:
            result = session.run(
                f"MATCH (n:{scientific_domain}) RETURN count(n) as count"
            )
            return result.single()["count"]

    def __del__(self):
        """Clean up Neo4j driver connection."""
        self._close()
