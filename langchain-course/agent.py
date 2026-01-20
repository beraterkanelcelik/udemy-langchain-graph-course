from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from tavily import TavilyClient
from langchain_tavily import TavilySearch

import os

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

tavily = TavilyClient(api_key=TAVILY_API_KEY)

@tool
def search(query: str) -> str:
    """Search the web for information
    Args:
        query: The query to search for
    Returns:
        The search result
    """
    print(f"Searching for: {query}")
    return tavily.search(query)

llm = ChatOpenAI()
#tools = [search]
tools = [TavilySearch()]
agent = create_agent(model=llm, tools=tools)
def main():
     #result = agent.invoke({"messages":HumanMessage(content="What is the capital of France?")})
     result = agent.invoke({"messages":HumanMessage(content="What is the capital of France?")})
     print(result)



if __name__ == "__main__":
    main()
