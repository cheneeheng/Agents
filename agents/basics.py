"""
Basics of using langgraph.

Based on: https://langchain-ai.github.io/langgraph/tutorials/introduction/
"""

import json
from typing import Annotated, Any

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from typing_extensions import TypedDict


class BasicToolNode:
    """A node that runs the tools requested in the last AIMessage."""

    def __init__(self, tools_: list) -> None:
        self.tools_by_name = {tool.name: tool for tool in tools_}

    def __call__(self, inputs: dict[str, Any]) -> dict[str, list[ToolMessage]]:
        if messages := inputs.get("messages", []):
            message = messages[-1]
        else:
            raise ValueError("No message found in input")
        outputs = []
        for tool_call in message.tool_calls:
            tool_result = self.tools_by_name[tool_call["name"]].invoke(
                tool_call["args"]
            )
            outputs.append(
                ToolMessage(
                    content=json.dumps(tool_result),
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"],
                )
            )
        return {"messages": outputs}


class State(TypedDict):  # pylint: disable=C0115
    messages: Annotated[list, add_messages]


graph_builder = StateGraph(State)


tool: BaseTool = TavilySearchResults(max_results=2)
tools = [tool]
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)
llm_with_tools = llm.bind_tools(tools)  # type: ignore


def chatbot(state: State):  # pylint: disable=C0116
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


graph_builder.add_node("chatbot", chatbot)

tool_node = ToolNode(tools=[tool])
graph_builder.add_node("tools", tool_node)

graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
)
# Any time a tool is called, we return to the chatbot to decide the next step
graph_builder.add_edge("tools", "chatbot")
graph_builder.set_entry_point("chatbot")
graph = graph_builder.compile()
