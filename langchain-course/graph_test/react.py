from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch



load_dotenv()

@tool
def triple(num: float) -> float:
    """
    param num: The number to triple
    return: The triple of the number
    """
    return num * 3


tools = [triple, TavilySearch(max_results=3)]

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0).bind_tools(tools)


