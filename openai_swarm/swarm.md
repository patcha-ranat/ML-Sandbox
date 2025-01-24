# Research on OpenAI Swarm

*Official GitHub - openai - swarm: https://github.com/openai/swarm?tab=readme-ov-file*

## What it is
- Orchestrator of AI agents (LLMs)
- Each agent can use a different model ***that are only available on OpenAI***
- Can be used though Python API as a client session
- Stateless
    - Can't work across multiple sessions
    - Reduce overhead of managing persistent states
    - High computational cost

## Advantages
- Minimal Codebase + Lightweight + The cleanest Multi-agent framework.
- It's proper as a starting point for learners to learn about Multi-agent frameworks, giving a sense of how multi-agent framework generally looks like.

## Disadvantages
- It's not mature as other tools, suited for multi-agent POC.
- It's for educational purposes (OpenAI does not actively support the framework).
- It's more for multi-agent prototyping.
- There's a cost from using OpenAI API for models utilization.

## Alternatives:
- **CrewAI**: *(The community claimed to be the most similar to OpenAI Swarm, but better and suited for production-scale LLM application)*
- **Langroid**: *(Claimed by library maintainer to be a better choice compared with Langgraph)*
- **Langgraph**: *(It's upgraded Langchain with cleaner workflows and visualization as a graph for agents working process)*
- **AutoGen**

## References:

- A glance of OpenAI Swarm
    - Narrative of what Swarm is, how it work underhood, and its perspective in the community
        - [Exploring OpenAI’s Swarm: An experimental framework for multi-agent systems - Medium](https://medium.com/@michael_79773/exploring-openais-swarm-an-experimental-framework-for-multi-agent-systems-5ba09964ca18)
    - Summary of Swarm and a few elaborated on API parameters + how code looks like
        - [Swarm: The Agentic Framework from OpenAI - Composio](https://composio.dev/blog/swarm-the-agentic-framework-from-openai/)
    - Opinions on how amature it is as a framework
        - [OpenAI Swarm: The Agentic Framework – Should You Care? - Reddit](https://www.reddit.com/r/LocalLLaMA/comments/1g56itb/openai_swarm_the_agentic_framework_should_you_care/?rdt=47947)
    - Swarm in abstraction and a few comparison between other multi-agent frameworks
        - [Swarm: The Agentic Framework from OpenAI - Fluid AI](https://www.fluid.ai/blog/swarm-the-agentic-framework-from-openai)
- Comparison between OpenAI Swarm and other multi-agent frameworks
    - Compared with code from different frameworks
        - [Choosing the Right AI Agent Framework: LangGraph vs CrewAI vs OpenAI Swarm - relari ](https://www.relari.ai/blog/ai-agent-framework-comparison-langgraph-crewai-openai-swarm)
    - Quick summary of other frameworks to choose the one for the right use case
        - [My thoughts on the most popular frameworks today: crewAI, AutoGen, LangGraph, and OpenAI Swarm - reddit](https://www.reddit.com/r/LangChain/comments/1g6i7cj/my_thoughts_on_the_most_popular_frameworks_today/)
    - Narrative in details for each framework
        - [AutoGen vs crewAI vs LangGraph vs OpenAI Swarm: Which AI Agent Framework Wins?](https://www.gettingstarted.ai/best-multi-agent-ai-framework/)
    - Langroid vs Langgraph
        - [Choosing Between Langroid and LangGraph for an E-commerce Chatbot Project - GitHub Langroid](https://github.com/langroid/langroid/discussions/390)
    - Overview of features and limitations for each framework
        - [Magentic-One, AutoGen, LangGraph, CrewAI, or OpenAI Swarm: Which Multi-AI Agent Framework is Best? - Medium](https://medium.com/data-science-in-your-pocket/magentic-one-autogen-langgraph-crewai-or-openai-swarm-which-multi-ai-agent-framework-is-best-6629d8bd9509)
- Swarm Code Example from Official Repository
    - [main.py, example on swarm github - GitHub openai/swarm](https://github.com/openai/swarm/blob/main/examples/support_bot/main.py)

## Pseudo Code: How it looks like
```python
# Import library
from swarm import Swarm, Agent


# Define Agents' Actions

def transfer_to_agent_b():
    """Empty function returning the other agent instance when it's called"""
    
    return AGENT_B

def transfer_to_agent_a():
    """Empty function returning the other agent instance when it's called"""
    
    return AGENT_A

def action_x():
    """Action can be something simple as return a string"""
    
    return {"response": "Hello World"}

def action_y():
    """Action can also be complex such as querying vector database utilizing RAG, or sending an email"""
    
    return {"response": ""}

def action_z(context: dict):
    """Action can also use framework's context to take action specific to conditions"""
    
    if context.get("user_attribute"):
        response = "Fizz"
    else:
        response = "Buzz"
    return {"response": response}


# Define Agents

AGENT_A = Agent(
    name="Agent A",
    instructions="Description about agent A's responsibility",
    functions=[
        transfer_to_agent_b, 
        action_x,
    ],
)

AGENT_B = Agent(
    name="Agent B",
    instructions="Description about agent B's responsibility",
    functions=[
        transfer_to_agent_a, 
        action_y, 
        action_z
    ],
)

# Process

client = Swarm() # required environment variable: 'OPENAI_API_KEY'

additional_context = {
    "user_name": "XXX",
    "user_attribute": True 
}

response = client.run(
    agent=AGENT_A,
    messages=[{"role": "user", "content": "User's prompt to specify requirement"}],
    context_variables=additional_context
)
```

## Demo + Result
```python
import os
from swarm import Swarm, Agent


# Define Context for Agent Instructions
additional_context = {
    "table_detail": """
- table_a is related to customer profile history, having column 'M' (Primary Key), 'N', and 'O'
- table_b is related to transactions of customer loan request, having column 'P' (Primary Key), 'Q' which is not useful in loan application, and 'R'
- table_c is not related to loan or customer detail
- table_d is missing which contain columns 'S' and 'T' that have meaningful meaning to business use case.
"""
}

# Define Actions

def transfer_to_leader():
    return agent_leader

def transfer_to_agent_domain_expert():
    return agent_domain_expert

def transfer_to_agent_data_engineer():
    return agent_data_engineer

# Define Agents

agent_leader = Agent(
    name="Agent Leader",
    instructions="""You are a helpful agent leading the project, and being an interface agent between Domain Expert Agent, Data Engineer Agent, and user.

    You only communicate and transfer user's requirement to agent that have relevant responsibility to that requirements and return response to user.

    You don't know about the given context.
    """,
    functions=[
        transfer_to_agent_domain_expert,
        transfer_to_agent_data_engineer,
    ],
)

agent_domain_expert = Agent(
    name="Agent Domain Expert",
    instructions=f"""You are Domain Expert Agent. You don't communicate to user and Data Engineer Agent directly, except through Agent Leader.

    You know very well about business domain and what information are needed for specific use cases.

    You know about the given context very well, but give useful answer to Agent Leader.

    Context: {additional_context.get('table_detail')}
    """,
    functions=[transfer_to_leader]
)

agent_data_engineer = Agent(
    name="Agent Data Engineer",
    instructions="""You are Data Engineer Agent. You don't communicate to user and Domain Expert Agent directly, except through Agent Leader.

    You know that if there's missing table or any other column than existing is required, you have to take time to ingest new data sources, and inform all stakeholder about timeline which take around 2 days.

    You don't know about the given context, but give useful answer to Agent Leader.
    """,
    functions=[transfer_to_leader]
)

# Process

os.environ["OPENAI_API_KEY"] = "[REDACTED]"

client = Swarm()

response = client.run(
    agent=agent_leader,
    messages=[{"role": "user", "content": "which tables and columns have useful information about loan application? and I need timeline if there's missing information."}],
    context_variables=additional_context
)

print(response.messages[-1]["content"])
```
### Result

```
Please wait while I gather the necessary information from the Domain Expert Agent regarding the tables and columns useful for loan applications. Additionally, I am coordinating the timeline for acquiring any missing information with the Data Engineer. 

The Domain Expert will provide the specific details needed, and if any additional data sourcing is required, the Data Engineer estimates it will take around 2 days to complete the ingestion process.
```

```python
response.messages

# [{'content': None,
#   'refusal': None,
#   'role': 'assistant',
#   'audio': None,
#   'function_call': None,
#   'tool_calls': [{'id': 'call_G8OuCntOaTxLgE6mx3nyDgFS',
#     'function': {'arguments': '{}', 'name': 'transfer_to_agent_domain_expert'},
#     'type': 'function'},
#    {'id': 'call_9LChBH27bVNJR27YQg17xOIQ',
#     'function': {'arguments': '{}', 'name': 'transfer_to_agent_data_engineer'},
#     'type': 'function'}],
#   'sender': 'Agent Leader'},
#  {'role': 'tool',
#   'tool_call_id': 'call_G8OuCntOaTxLgE6mx3nyDgFS',
#   'tool_name': 'transfer_to_agent_domain_expert',
#   'content': '{"assistant": "Agent Domain Expert"}'},
#  {'role': 'tool',
#   'tool_call_id': 'call_9LChBH27bVNJR27YQg17xOIQ',
#   'tool_name': 'transfer_to_agent_data_engineer',
#   'content': '{"assistant": "Agent Data Engineer"}'},
#  {'content': 'Please wait while I gather the necessary information from the Domain Expert Agent regarding the tables and columns useful for loan applications. Additionally, I am coordinating the timeline for acquiring any missing information with the Data Engineer. \n\nThe Domain Expert will provide the specific details needed, and if any additional data sourcing is required, the Data Engineer estimates it will take around 2 days to complete the ingestion process.',
#   'refusal': None,
#   'role': 'assistant',
#   'audio': None,
#   'function_call': None,
#   'tool_calls': None,
#   'sender': 'Agent Data Engineer'}]
```

The framework still did not work as expected by the following detail:
1. Not returning answer from Agents to User.
2. Sender should be Leader Agent, not Data Engineer Agent according to prompts.

---
