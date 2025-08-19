import os
from typing import Literal
from dotenv import load_dotenv

from tavily import TavilyClient
from deepagents import create_deep_agent, SubAgent

# Import ChatGroq directly from langchain_groq (keeping existing setup)
from langchain_groq import ChatGroq

# Use absolute imports instead of relative imports
from src.prompts import (
    SUB_RESEARCH_PROMPT,
    SUB_CRITIQUE_PROMPT,
    RESEARCH_INSTRUCTIONS,
)

# Load environment variables from .env file
load_dotenv()

# Initialize Kimi-K2-Instruct model through Groq (best approach!)
kimi_model = ChatGroq(
    model="moonshotai/kimi-k2-instruct",  # Kimi model available through Groq
    api_key=os.environ["GROQ_API_KEY"],  # Same Groq API key
    temperature=0.1,  # Lower temperature for more focused research
    max_tokens=4096,  # Adjust based on your needs (model supports 131k context!)
)

# Keep existing TavilyClient setup
tavily_client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])

def internet_search(
    query: str,
    max_results: int = 5,
    topic: Literal["general", "news", "finance"] = "general",
    include_raw_content: bool = False,
):
    """Run a web search"""
    search_docs = tavily_client.search(
        query,
        max_results=max_results,
        include_raw_content=include_raw_content,
        topic=topic,
    )
    return search_docs

# Sub-agent configurations using imported prompts
research_sub_agent = {
    "name": "research-agent",
    "description": "Used to research more in depth questions. Only give this researcher one topic at a time. Do not pass multiple sub questions to this researcher. Instead, you should break down a large topic into the necessary components, and then call multiple research agents in parallel, one for each sub question.",
    "prompt": SUB_RESEARCH_PROMPT,
    "tools": ["internet_search"]
}

critique_sub_agent = {
    "name": "critique-agent",
    "description": "Used to critique the final report. Give this agent some information about how you want it to critique the report.",
    "prompt": SUB_CRITIQUE_PROMPT,
}

# Create the agent using Kimi-K2-Instruct model (via Groq) and imported research instructions
agent = create_deep_agent(
    [internet_search],
    RESEARCH_INSTRUCTIONS,
    model=kimi_model,  # Pass the initialized Kimi model here
    subagents=[critique_sub_agent, research_sub_agent],
).with_config({"recursion_limit": 1000})
