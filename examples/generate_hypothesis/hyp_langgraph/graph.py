from typing import Literal

from langgraph.graph import END, START, StateGraph
from langgraph.graph.graph import CompiledGraph
from loguru import logger

from .agents.analysts import create_analyst_agent
from .agents.critique_analyst import create_critique_analyst_agent
from .agents.hypothesis_generator import create_hypothesis_generator_agent
from .agents.hypothesis_refiner import create_hypothesis_refiner_agent
from .agents.ontologist import create_ontologist_agent
from .agents.summary import create_summary_agent
from .state import HypgenState


def improve_hypothesis(
    state: HypgenState,
) -> Literal["hypothesis_refiner", "summary_agent"]:
    if state["iteration"] > 3:
        logger.info("Iteration limit reached after {} iterations", state["iteration"])
        return "summary_agent"
    if "ACCEPT" in state["critique"]:
        logger.info("Hypothesis accepted after {} iterations", state["iteration"])
        return "summary_agent"
    else:
        logger.info("Hypothesis rejected after {} iterations", state["iteration"])
        return "hypothesis_refiner"


def create_hypgen_graph() -> CompiledGraph:
    graph = StateGraph(HypgenState)

    # Add nodes with specialized agents
    graph.add_node("ontologist", create_ontologist_agent("large")["agent"])
    graph.add_node(
        "hypothesis_generator", create_hypothesis_generator_agent("reasoning")["agent"]
    )
    graph.add_node(
        "hypothesis_refiner", create_hypothesis_refiner_agent("reasoning")["agent"]
    )
    graph.add_node(
        "novelty_analyst", create_analyst_agent("novelty", "reasoning")["agent"]
    )
    graph.add_node(
        "feasibility_analyst", create_analyst_agent("feasibility", "reasoning")["agent"]
    )
    graph.add_node(
        "impact_analyst", create_analyst_agent("impact", "reasoning")["agent"]
    )
    graph.add_node("critique_analyst", create_critique_analyst_agent("large")["agent"])
    graph.add_node("summary_agent", create_summary_agent("large")["agent"])

    # Add edges
    graph.add_edge(START, "ontologist")
    graph.add_edge("ontologist", "hypothesis_generator")
    # # Fork initial hypothesis
    graph.add_edge("hypothesis_generator", "novelty_analyst")
    graph.add_edge("hypothesis_generator", "feasibility_analyst")
    graph.add_edge("hypothesis_generator", "impact_analyst")
    # # Fork refined hypothesis
    graph.add_edge("hypothesis_refiner", "novelty_analyst")
    graph.add_edge("hypothesis_refiner", "feasibility_analyst")
    graph.add_edge("hypothesis_refiner", "impact_analyst")
    # # Join
    graph.add_edge("novelty_analyst", "critique_analyst")
    graph.add_edge("feasibility_analyst", "critique_analyst")
    graph.add_edge("impact_analyst", "critique_analyst")
    # graph.add_edge("critique_analyst", END)
    graph.add_conditional_edges(
        "critique_analyst",
        improve_hypothesis,
    )
    graph.add_edge("summary_agent", END)

    return graph.compile()
