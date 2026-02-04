from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from dotenv import load_dotenv
from schemas import AgentResponse

load_dotenv()
llm = ChatOpenAI(model="gpt-4o-mini")
tools = [TavilySearch()]



agent = create_agent(llm, tools=tools, response_format=AgentResponse)



if __name__ == "__main__":
    result = agent.invoke({
    "messages": [
        {
            "role": "user",
            "content": "bring me the latest news about the stock market"
        }
        ]
    })

    print(result["structured_response"])