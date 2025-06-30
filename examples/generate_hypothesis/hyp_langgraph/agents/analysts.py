import os
from typing import Any, Dict, Literal, Optional

from langchain.prompts import PromptTemplate
from langchain_community.tools.arxiv.tool import ArxivQueryRun
from langchain_community.tools.pubmed.tool import PubmedQueryRun
from langchain_community.utilities.pubmed import PubMedAPIWrapper
from langgraph.prebuilt.chat_agent_executor import create_react_agent
from loguru import logger

from ..llm.utils import get_model
from ..state import HypgenState
from ..tools.perplexity import search_perplexity
from ..utils import add_role

SEARCH_PROMPT = """
You are a research assistant. Find relevant literature that will help evaluate {analysis_type} of a hypothesis given below.

To find a broad range of relevant literature, use queries that are related to the hypothesis but not too specific.
Instead of one or two very specific queries, use a broader range of queries that are related to the hypothesis.

Hypothesis:
{hypothesis}

After searching, return the search results.
"""

ANALYST_PROMPT = """
You are a critical AI assistant collaborating with a group of scientists to assess the {analysis_type} of a research proposal. 

Your primary task is to evaluate a proposed research hypothesis for its {task}

After careful analysis, return your estimations for the {analysis_type} as one of the following:
- "Not {analysis_type}"
- "Somewhat {analysis_type}"
- "{analysis_type}"
- "Very {analysis_type}"

Provide your reasoning for your assessment.
Cite the literature to support your assessment.

Literature information gathered by the research assistant:
{literature}

Hypothesis:
{hypothesis}
"""

# "Your primary task is to evaluate a proposed research hypothesis for its ..."
PROMPT_TASK_TEMPLATES = {
    "novelty": """
    novelty ensuring it does not overlap significantly with existing literature or delve into areas that are already well-explored.
    """,
    "feasibility": """
    feasibility considering the resources, time, and technical challenges.
    """,
    "impact": """
    impact considering the potential scientific, technological, and societal impact.
    """,
}


tools = [
    ArxivQueryRun(),
    PubmedQueryRun(api_wrapper=PubMedAPIWrapper(api_key=os.getenv("PUBMED_API_KEY"))),
    search_perplexity,
]


def create_analyst_agent(
    analyst: Literal["novelty", "feasibility", "impact"],
    model: Optional[Literal["large", "small", "reasoning"]] = None,
    **kwargs,
) -> Dict[str, Any]:
    """Creates a analyst agent that evaluates the hypothesis."""

    analyst_prompt = PromptTemplate.from_template(ANALYST_PROMPT)

    llm = get_model(model, **kwargs)
    chain = analyst_prompt | llm
    research_assistant = create_react_agent(model=llm, tools=tools)

    def agent(state: HypgenState) -> HypgenState:
        logger.info(f"Starting {analyst} analysis")
        # Search for literature
        messages = (
            PromptTemplate.from_template(SEARCH_PROMPT)
            .invoke({"analysis_type": analyst, "hypothesis": state["hypothesis"]})
            .to_messages()
        )
        logger.info("Searching for relevant literature")
        assistant_response = research_assistant.invoke({"messages": messages})
        logger.info("Literature search completed")

        # Get the literature information from the response
        literature = assistant_response["messages"][-1]

        # Run the chain
        logger.info(f"Running {analyst} analysis chain")
        response = chain.invoke(
            {
                **state,
                "literature": literature,
                "analysis_type": analyst,
                "task": PROMPT_TASK_TEMPLATES[analyst],
            }
        )
        logger.info(f"{analyst} analysis completed successfully")

        return {
            "messages": assistant_response["messages"] + [add_role(response, analyst)],
            analyst: response.content,
        }

    return {"agent": agent}
