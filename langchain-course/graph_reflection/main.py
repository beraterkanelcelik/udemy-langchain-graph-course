from dotenv import load_dotenv
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from chains import generation_chain, reflection_chain


load_dotenv()

class MessageGraph(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


REFLECT = "reflect"
GENERATE = "generate"

def generation_node(state: MessageGraph):
    return {"messages": [generation_chain.invoke({"messages": state["messages"]})]}

def reflection_node(state: MessageGraph):
    res = reflection_chain.invoke({"messages": state["messages"]})
    return {"messages": [HumanMessage(content=res.content)]}

def should_continue(state: MessageGraph):
    if len(state["messages"]) > 6:
        return END
    return REFLECT

builder = StateGraph(MessageGraph)

builder.add_node(GENERATE, generation_node)
builder.add_node(REFLECT, reflection_node)

builder.set_entry_point(GENERATE)
builder.add_conditional_edges(GENERATE, should_continue, {
    END: END,
    REFLECT: REFLECT
})
builder.add_edge(REFLECT, GENERATE)


workflow = builder.compile()
print(workflow.get_graph().draw_mermaid_png(output_file_path="reflection_graph.png"))



















if __name__ == "__main__":
    inputs = HumanMessage(content="""
    Make this tweet better:
    @Claude Opus 4.6 is out, and it's amazing!
    """)
    res = workflow.invoke({"messages": [inputs]})
    print(res["messages"][-1].content)