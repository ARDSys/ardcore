from langgraph.graph import MessagesState


class HypgenState(MessagesState):
    subgraph: str
    context: str
    hypothesis: str

    novelty: str
    feasibility: str
    impact: str

    critique: str
    summary: str

    iteration: int
