from typing import TYPE_CHECKING, Any, Protocol

from ard.subgraph import Subgraph

if TYPE_CHECKING:
    from ard.hypothesis.hypothesis import Hypothesis


class HypothesisGeneratorProtocol(Protocol):
    def run(self, subgraph: Subgraph) -> "Hypothesis": ...

    def __str__(self) -> str: ...

    def to_json(self) -> dict[str, Any]: ...
