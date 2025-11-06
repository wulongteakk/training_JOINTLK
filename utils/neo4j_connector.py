from __future__ import annotations

from collections import deque
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

try:  # pragma: no cover - optional dependency runtime check
    import pyahocorasick  # type: ignore
    _PYAHO_AVAILABLE = True
except ModuleNotFoundError:  # pragma: no cover - fallback branch
    _PYAHO_AVAILABLE = False

    class _FallbackAutomaton:
        """Minimal substring matcher used when ``pyahocorasick`` is missing.

        The fallback keeps an in-memory list of surface forms and performs a
        naive ``str.find`` scan over the input text.  Although this is slower
        than the automaton-based approach, it preserves the same public API so
        that downstream code can continue to call ``iter`` and receive
        ``(end_index, value)`` tuples.
        """

        def __init__(self) -> None:
            self._keywords: List[Tuple[str, str]] = []

        def add_word(self, word: str, value: str) -> None:
            self._keywords.append((word, value))

        def make_automaton(self) -> None:  # noqa: D401 - interface parity
            """No-op to mirror :mod:`pyahocorasick` initialisation."""

        def iter(self, text: str):  # type: ignore[override]
            for word, value in self._keywords:
                start = 0
                while True:
                    idx = text.find(word, start)
                    if idx == -1:
                        break
                    yield idx + len(word) - 1, value
                    start = idx + 1

    class _PyAhoNamespace:
        Automaton = _FallbackAutomaton

    pyahocorasick = _PyAhoNamespace()  # type: ignore
    print(
        "[neo4j_connector] pyahocorasick module not found; "
        "falling back to a slow Python matcher. Install 'pyahocorasick' "
        "for optimal performance."
    )

from neo4j import GraphDatabase


class Neo4jConnector:
    """
    用于连接 Neo4j、执行实体链接和动态获取子图的类。
    """

    def __init__(
        self,
        uri: str,
        user: str,
        password: str,
        grounding_vocab_path: str,
        *,
        entity_property: str = "name",
    ) -> None:
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.entity_property = entity_property
        self.automaton = pyahocorasick.Automaton()
        self._load_grounding_vocabulary(grounding_vocab_path)

        # ------------------------------------------------------------------
        # Grounding helpers
        # ------------------------------------------------------------------
        def _load_grounding_vocabulary(self, grounding_vocab_path: str) -> None:
            """Populate the Aho–Corasick automaton with entity mentions."""

            loaded = 0
            with open(grounding_vocab_path, "r", encoding="utf8") as fin:
                for idx, line in enumerate(fin):
                    mention = line.strip()
                    if mention:
                        self.automaton.add_word(mention, mention)
                        loaded += 1

            self.automaton.make_automaton()
            print(f"Loaded {loaded} surface forms from {grounding_vocab_path}.")

        def link_entities(self, query: str) -> List[str]:
            """Return the unique entity mentions detected inside ``query``."""

            return sorted({match for _, match in self.automaton.iter(query)})

        # ------------------------------------------------------------------
        # Neo4j traversal helpers
        # ------------------------------------------------------------------
        def close(self) -> None:
            self.driver.close()

        def _record_to_node_payload(self, node: Any) -> Dict[str, Any]:
            return {
                "id": node.element_id,
                "labels": list(node.labels),
                "properties": dict(node),
            }

        def _record_to_edge_payload(self, relationship: Any) -> Dict[str, Any]:
            return {
                "id": relationship.element_id,
                "type": relationship.type,
                "start_node_id": relationship.start_node.element_id,
                "end_node_id": relationship.end_node.element_id,
                "properties": dict(relationship),
            }

        def fetch_subgraph_for_entities(
                self,
                entities: Sequence[str],
                *,
                hops: int = 1,
        ) -> Dict[str, Any]:
            """Fetch an ``hops``-hop subgraph around ``entities``.

            The method performs a breadth-first traversal starting from all nodes
            whose ``entity_property`` matches any of the supplied surface forms.  It
            returns a dictionary with two keys:

            ``nodes``
                A list of dictionaries containing the Neo4j ``element_id`` of the
                node, its labels, and all properties.
            ``edges``
                A list describing the traversed relationships.  Each element stores
                the ``element_id`` of the relationship and the identifiers of the
                incident nodes.
            ``mention_node_ids``
                The subset of node ``element_id`` values that correspond to the
                grounded entities.  This is convenient when the caller wants to add
                special edges from the synthetic context node.
            """

            if not entities:
                return {"nodes": [], "edges": [], "mention_node_ids": []}

            entity_set: Set[str] = set(entities)
            nodes: Dict[str, Dict[str, Any]] = {}
            edges: List[Dict[str, Any]] = []
            mention_node_ids: Set[str] = set()

            with self.driver.session() as session:
                # Retrieve all seed nodes first.
                seed_records = session.run(
                    f"""
                        MATCH (seed)
                        WHERE seed.{self.entity_property} IN $entities
                        RETURN seed
                        """,
                    entities=list(entity_set),
                )

                queue: deque = deque()
                for record in seed_records:
                    seed_node = record["seed"]
                    node_payload = self._record_to_node_payload(seed_node)
                    nodes[node_payload["id"]] = node_payload
                    queue.append((node_payload["id"], 0))
                    if node_payload["properties"].get(self.entity_property) in entity_set:
                        mention_node_ids.add(node_payload["id"])

                visited: Set[str] = set()

                while queue:
                    current_id, depth = queue.popleft()
                    if current_id in visited or depth >= hops:
                        visited.add(current_id)
                        continue
                    visited.add(current_id)

                    records = session.run(
                        """
                        MATCH (seed)-[rel]-(neighbor)
                        WHERE elementId(seed) = $seed_id
                        RETURN seed, rel, neighbor
                        """,
                        seed_id=current_id,
                    )

                    for record in records:
                        rel = record["rel"]
                        neighbor = record["neighbor"]

                        neighbor_payload = self._record_to_node_payload(neighbor)
                        neighbor_id = neighbor_payload["id"]
                        if neighbor_id not in nodes:
                            nodes[neighbor_id] = neighbor_payload
                            queue.append((neighbor_id, depth + 1))
                            if neighbor_payload["properties"].get(self.entity_property) in entity_set:
                                mention_node_ids.add(neighbor_id)

                        edges.append(self._record_to_edge_payload(rel))

            return {
                "nodes": list(nodes.values()),
                "edges": edges,
                "mention_node_ids": list(mention_node_ids),
            }


__all__ = ["Neo4jConnector"]



