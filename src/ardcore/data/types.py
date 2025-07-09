from typing import Callable, List, TypedDict


class Triplet(TypedDict):
    node_1: str
    edge: str
    node_2: str


KGGenerator = Callable[[str], List[Triplet]]
