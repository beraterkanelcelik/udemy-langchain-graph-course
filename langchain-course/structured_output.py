from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from tavily import TavilyClient
from langchain_tavily import TavilySearch
from typing import List
from pydantic import BaseModel, Field


import os

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")


class Source(BaseModel):
    """Schema for a source used by the agent"""

    url:str = Field(description="The url of the source")

class AgentResponse(BaseModel):
    """Schema for the agent's response"""
    answer:str = Field(description="The answer to the question")
    sources:List[Source] = Field(default_factory=list, description="The sources used to answer the question")

llm = ChatOpenAI()
tools = [TavilySearch()]
agent = create_agent(model=llm, tools=tools, response_format=AgentResponse)





def main():
     #result = agent.invoke({"messages":HumanMessage(content="What is the capital of France?")})
     result = agent.invoke({"messages":HumanMessage(content="give me 3 jobs from linkedin.com that are related to AI and machine learning")})
     print(result)



if __name__ == "__main__":
    main()
