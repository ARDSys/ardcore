# 📖 Complete Technical Guide

Your comprehensive guide to building ARD community extensions - from core concepts to implementation to contribution.

## Table of Contents
1. [Understanding Core Concepts](#understanding-core-concepts)
2. [Environment Setup](#environment-setup)
3. [Implementation Guide](#implementation-guide)
4. [Building Your Extension](#building-your-extension)
5. [Contributing](#contributing)
6. [FAQ](#frequently-asked-questions)

---

## Understanding Core Concepts

### The Subgraph (`ard.subgraph.subgraph.Subgraph`)

Think of a vast Knowledge Graph (KG) containing interconnected scientific concepts, findings, and relationships extracted from literature. A `Subgraph` is a small, focused section of this larger KG, representing an interesting or potentially novel path between two or more concepts.

#### How are Subgraphs Created?

Imagine an **"explorer" agent** journeying through a vast knowledge graph (KG), navigating connections between concepts like a scientist wandering through an interconnected landscape of ideas. As it traverses the KG, the path it follows forms what we call a **Subgraph**.

We've developed **custom traversal algorithms** that empower these agents to independently explore the graph, uncovering **unique and creative paths** through scientific domains. These subgraphs aren't just random—they're shaped by the agent's internal knowledge, learned patterns, and a touch of stochasticity.

#### Why Subgraphs?

Subgraphs serve as targeted prompts or creative seeds for your Multi-Agent System (MAS). Rather than navigating the entire knowledge graph (KG), your agents can zoom in on a focused cluster of concepts and relationships. Each subgraph distills a slice of the KG into a creative spark—activating the latent knowledge within Large Language Models and guiding agents toward novel scientific insights.

#### Key `Subgraph` Attributes:
*   `start_node`: The starting concept of the path.
*   `end_node`: The ending concept of the path.
*   `path_nodes`: The list of concepts forming the direct path.
*   `get_path_edges()`: Returns the relationships (edges) along the direct path.
*   `to_cypher_string()`: Provides a textual representation of the subgraph's nodes and relationships, useful for LLM prompts.
*   `contextualize()`: (Optional) Can be used to generate an LLM-based analysis of the subgraph's content, providing richer context.
*   It inherits from `KnowledgeGraph`, so you can use methods like `get_nodes()`, `get_edges()`, `get_node_attrs()`, etc.

#### Example Subgraph

```text
Subgraph(start="Inflammation", end="Alzheimer's Disease", path_length=3)

Path: Inflammation -> increases -> Amyloid Beta -> accumulates in -> Alzheimer's Disease

Additional nodes: Microglia, Tau Protein, Neuroinflammation
Additional relationships: 
- Inflammation -> activates -> Microglia
- Microglia -> produces -> Neuroinflammation
- Neuroinflammation -> promotes -> Tau Protein
- Tau Protein -> contributes to -> Alzheimer's Disease
```

### The Hypothesis (`ard.hypothesis.hypothesis.Hypothesis`)

This is the desired output of your MAS. A `Hypothesis` object encapsulates:
*   `title`: A concise title for the hypothesis.
*   `statement`: The core research hypothesis statement.
*   `source`: A reference back to the `Subgraph` that inspired it.
*   `method`: A reference to the `HypothesisGeneratorProtocol` implementation that created it.
*   `references`: A list of scientific references supporting the hypothesis.
*   `metadata`: A dictionary for any additional information (e.g., agent names, confidence scores, intermediate steps).

#### Example Hypothesis

```python
hypothesis = Hypothesis(
    title="Microglial-Mediated Neuroinflammation as a Link Between Systemic Inflammation and Alzheimer's Pathology",
    statement="Systemic inflammation activates microglia, leading to neuroinflammation that promotes both amyloid beta accumulation and tau pathology, accelerating Alzheimer's disease progression.",
    source=subgraph,  # The original subgraph object
    method=your_generator,  # Your generator implementation
    references=[
        "Smith et al. (2019). Neuroinflammation and Neurodegeneration. Journal of Neuroscience, 40(1), 123-145.",
        "Chen, J. & Wong, T. (2021). Microglial Activation in Alzheimer's Disease. Nature Reviews Neuroscience, 22(4), 210-228."
    ],
    metadata={
        "confidence": 0.85,
        "generated_by": "YourTeamName MAS",
        "agent_contributions": {
            "research_agent": "Identified the microglial activation pathway",
            "critic_agent": "Suggested including tau pathology connection"
        }
    }
)
```

### Scientific Inspiration

Our approach draws inspiration from several pioneering efforts:

- **Sakana AI's *AI Scientist*** envisions fully automated researchers, achieving the first peer-reviewed publication authored by an AI scientist.
- **SciAgents** explored random path traversal through knowledge graphs to discover unexplored research directions.
- **Google's Co-Scientist project** highlights the power of extended deliberation via increased test-time compute.

---

## Environment Setup

Set up your development environment using UV:

```bash
# Clone the ARD repository
git clone https://github.com/your-repo/ard
cd ard
uv init
uv pip install -e .
# Make sure your API keys are set in a .env file (see .env.example)
```

---

## Implementation Guide

### Implement the `HypothesisGeneratorProtocol`

The only strict requirement is creating a Python class that implements the `HypothesisGeneratorProtocol`:

```python
# Located in: src/ard/hypothesis/types.py

from typing import Protocol, Any
from ard.subgraph import Subgraph

class HypothesisGeneratorProtocol(Protocol):
    def run(self, subgraph: Subgraph) -> "Hypothesis": ...
    def __str__(self) -> str: ... # For identifying your method
    def to_json(self) -> dict[str, Any]: ... # For serialization
```

### Minimal Working Example

```python
from ard.hypothesis import Hypothesis
from ard.hypothesis.types import HypothesisGeneratorProtocol
from ard.subgraph import Subgraph
from dataclasses import dataclass
from typing import Any, Dict

@dataclass
class SimpleHypothesisGenerator(HypothesisGeneratorProtocol):
    """A simple hypothesis generator using a single LLM call."""
    
    llm: Any  # Your LLM client
    
    def run(self, subgraph: Subgraph) -> Hypothesis:
        # Convert subgraph to string representation for the LLM
        graph_text = subgraph.to_cypher_string()
        
        # Create a prompt for the LLM
        prompt = f"""
        Based on the following knowledge graph:
        
        {graph_text}
        
        Generate a scientific hypothesis that explains the relationship between 
        {subgraph.start_node} and {subgraph.end_node}.
        
        Provide your response in this format:
        TITLE: [concise title for the hypothesis]
        HYPOTHESIS: [detailed hypothesis statement]
        REFERENCES: [list of references that support this hypothesis]
        """
        
        # Get response from LLM
        response = self.llm(prompt)
        
        # Parse response
        title_line = response.split("TITLE:")[1].split("HYPOTHESIS:")[0].strip()
        hypothesis_statement = response.split("HYPOTHESIS:")[1].split("REFERENCES:")[0].strip()
        
        # Parse references (if provided)
        references = []
        if "REFERENCES:" in response:
            references_text = response.split("REFERENCES:")[1].strip()
            references = [ref.strip() for ref in references_text.split("\n") if ref.strip()]
        
        # Create and return Hypothesis object
        return Hypothesis(
            title=title_line,
            statement=hypothesis_statement,
            source=subgraph,
            method=self,
            references=references,
            metadata={"generator": "SimpleHypothesisGenerator"}
        )
    
    def __str__(self) -> str:
        return "SimpleHypothesisGenerator"
    
    def to_json(self) -> Dict[str, Any]:
        return {
            "name": str(self),
            "type": "simple_llm_generator"
        }
```

### Implementation Tips

**Leverage Subgraph Data:** Use the nodes, edges, and context effectively. The `to_cypher_string()` method is helpful for prompts.

**Agent Roles:** Consider different roles: generating ideas, criticizing, refining, searching for supporting evidence, ensuring clarity.

**Prompt Engineering:** Craft clear and effective prompts for your LLM-powered agents.

**Tools Integration:** Give agents tools (database searches, web searches, biomedical APIs).

---

## Building Your Extension

We encourage creativity! You can approach building your extension in several ways:

### 1. Use the Sample Template
- The `sample_code/` directory provides a minimal structure implementing the `HypothesisGeneratorProtocol`. Use this as a clean slate.

### 2. Build From Scratch
- Use any MAS framework (CrewAI, AutoGen, LangGraph, Camel-AI) or build your own agent orchestration logic.

### 3. Modify Existing Extension
- Explore the `beehealthy/` directory - **our first community extension** showcasing sophisticated multi-agent collaboration.
- **Ideas:** Improve prompts, add new specialized agents, integrate new tools, refine agent interaction logic.

### Development Workflow

1. **Create your extension directory**: `community_extensions/your_extension_name/`
2. **Implement the protocol**: Follow the implementation guide above
3. **Test your extension**: Generate hypotheses and verify outputs
4. **Document your approach**: Create a README.md explaining your extension
5. **Submit your contribution**: Follow the contributing steps below

---

## Contributing

### Contributing Your Extension

1. **Fork the Repository**: Create your own fork of the ARD repository
2. **Create Your Extension**: In the `community_extensions/` directory, create a new folder with your extension name
3. **Documentation**: Include a README.md explaining your approach and how to use it
4. **Submit a Pull Request**: Submit a PR including:
   - Your implementation code
   - Your README.md documentation
   - Example outputs demonstrating your system's capabilities
   - **Detailed logs** showing agent interactions and decision processes

### Extension Checklist

Before submitting, ensure your extension:
- ✅ Has clear README.md with setup instructions
- ✅ Includes `.env.example` with required API keys  
- ✅ Lists all dependencies
- ✅ Runs without manual intervention
- ✅ Includes logging for transparency
- ✅ Is reproducible by others

---

## Frequently Asked Questions

### 🧠 LLMs & Technology

**Q: Do I need to use a specific LLM provider?**  
A: No, you can use any LLM provider (OpenAI, Anthropic, Cohere, etc.) as well as open-source models.

**Q: Can I use multiple different LLMs in my extension?**  
A: Yes, you can use different models for different agents or tasks.

**Q: Are there any restrictions on the technology I can use?**  
A: None at all. You're free to use any technology in the name of science.

**Q: Can I modify any part of the ARD codebase?**  
A: For community extensions, please focus on implementing your solution within the existing framework rather than modifying the core code.


### 🛠️ Development

**Q: How do I test my extension?**  
A: Generate hypotheses using your system and verify that they follow the `Hypothesis` object structure. Start with the `sample_code/` template to understand the basic structure.

**Q: What should I include in my extension's README?**  
A: Include installation instructions, usage examples, dependencies, and a description of your approach. See the `sample_code/` template for a basic structure, or BeeHealthy for a comprehensive example.

**Q: How do I handle API keys securely?**  
A: Include a `.env.example` file with dummy values showing required keys, and ensure your extension reads from environment variables.

**Q: Can I use external tools and APIs?**  
A: Absolutely! Many extensions benefit from integrating external knowledge sources, web search, or specialized biomedical APIs.

### 🤝 Community

**Q: How do I get support if I'm stuck?**  
A: Start with the `sample_code/` template, examine BeeHealthy for advanced examples, and consider opening an issue in the repository for complex questions.

---

## Examples

- **Simple Template:** See `sample_code/` for a minimal starting implementation
- **First Community Extension:** Explore `beehealthy/` - our inaugural extension showcasing sophisticated multi-agent collaboration

## References

* [1] [The AI Scientist: Towards Fully Automated Open-Ended Scientific Discovery](https://arxiv.org/abs/2408.06292)
* [2] [SciAgents: Automating scientific discovery through multi-agent intelligent graph reasoning](https://arxiv.org/abs/2409.05556)
* [3] [Towards an AI co-scientist](https://arxiv.org/abs/2502.18864) 