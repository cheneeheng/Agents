"""
Basics of using langgraph.

Reference:
https://langchain-ai.github.io/langgraph/tutorials/get-started/2-add-tools/
"""

import json
from typing import Annotated, Any, Literal

import dotenv
import requests  # type: ignore[import-untyped]
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import END, START
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import Command, interrupt
from typing_extensions import TypedDict

dotenv.load_dotenv()


# =============================================================================
# Graph - State
# =============================================================================
class State(TypedDict):  # pylint: disable=C0115
    messages: Annotated[list, add_messages]


# =============================================================================
# Graph - Router Node
# =============================================================================
def route_tools(state: State):
    """
    Use in the conditional_edge to route to the ToolNode if the last message
    has tool calls. Otherwise, route to the end.
    """
    if isinstance(state, list):
        ai_message = state[-1]  # pyright: ignore[reportArgumentType]
    elif messages := state.get("messages", []):
        ai_message = messages[-1]
    else:
        raise ValueError(
            f"No messages found in input state to tool_edge: {state}"
        )
    if (
        hasattr(ai_message, "tool_calls")
        and len(ai_message.tool_calls) > 0  # type: ignore[arg-type]
    ):
        return "tools"
    return END


# =============================================================================
# Custom Tools
# =============================================================================
@tool("multiply_tool", parse_docstring=True)
def multiply(a: int, b: int) -> int:
    """Multiply two numbers.

    Args:
        a: First operand
        b: Second operand
    """
    return a * b * 2


@tool("multiplizieren_tool", parse_docstring=True)
def multiplizieren(a: int, b: int) -> int:
    """Zwei Zahlen multiplizieren.

    Args:
        a: Erster Operand
        b: Zweiter Operand
    """
    return a * b


@tool("check_website_status_tool", parse_docstring=True)
def check_website_status(url: str) -> str:
    """Check the status of a website.

    Args:
        url: Url of a website
    """
    url_status_code = requests.get(
        url, allow_redirects=True, timeout=30
    ).status_code
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-lite",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
    )
    response = llm.invoke(
        [
            {
                "role": "user",
                "content": (
                    f"A website has status code : {url_status_code}. "
                    f"What is the status of the website in 1 sentence?"
                ),
            }
        ]  # type: ignore[return-value]
    )

    return response.content  # type: ignore[return-value]  # pyright: ignore[reportReturnType]


tools = [
    multiply,
    multiplizieren,
    check_website_status,
]


# =============================================================================
# Graph - Nodes
# =============================================================================
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


def human_approval(
    state: State,
) -> Command[Literal[END, "chatbot"]]:  # pyright: ignore[reportInvalidTypeForm]
    """A node that asks for human approval if the answer is correct.

    Args:
        state (State): graph state

    Returns:
        Command[Literal[END, "chatbot"]]: _description_
    """
    is_approved = interrupt(
        {
            "question": "Is this correct? (Y/N)",
        }
    )

    if is_approved == "N":
        return Command(
            goto="chatbot",
            update={
                "messages": [
                    HumanMessage(
                        content="The answer is wrong, use another tool."
                    )
                ]
            },
        )
    else:
        return Command(
            goto=END,
            update={
                "messages": [
                    HumanMessage(
                        content=(
                            f"Yes, {state['messages'][-1].content} "
                            f"is correct"
                        )
                    )
                ]
            },
        )


# =============================================================================
# Graph
# =============================================================================
def build_graph(memory_: MemorySaver) -> CompiledStateGraph:
    """Builds the graph.

    Args:
        memory_ (MemorySaver): Memory for the graph.

    Returns:
        CompiledStateGraph: Compiled graph.
    """

    # -------------------------------------------------------------------------
    # Create LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-lite",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
    )
    llm_with_tools = llm.bind_tools(tools)  # type: ignore[arg-type]

    # -------------------------------------------------------------------------
    # Create graph
    graph_builder = StateGraph(State)

    # --------------------
    # Create graph - Add nodes

    def chatbot(state: State):  # pylint: disable=C0116
        return {"messages": [llm_with_tools.invoke(state["messages"])]}

    graph_builder.add_node("chatbot", chatbot)

    tool_node = BasicToolNode(tools_=tools)
    # OR
    # tool_node = ToolNode(tools=[tool])
    graph_builder.add_node("tools", tool_node)

    graph_builder.add_node("human_approval", human_approval)

    # --------------------
    # Create graph - Add edges

    graph_builder.add_edge(START, "chatbot")
    # OR
    # graph_builder.set_entry_point("chatbot")

    # Returns "tools" if the chatbot asks to use a tool, and "END" if it is
    # fine directly responding. This conditional routing defines the main
    # agent loop.
    graph_builder.add_conditional_edges(
        "chatbot",
        route_tools,
        # The following dictionary lets you tell the graph to interpret the
        # condition's outputs as a specific node. It defaults to the identity
        # function, but if you want to use a node named something else apart
        # from "tools", You can update the value of the dictionary to
        # something else e.g., "tools": "my_tools"
        {"tools": "tools", END: END},
    )
    # OR
    # graph_builder.add_conditional_edges(
    #     "chatbot",
    #     tools_condition,
    # )

    # Any time a tool is called, ask the human for approval.
    graph_builder.add_edge("tools", "human_approval")

    graph = graph_builder.compile(checkpointer=memory_)

    return graph


if __name__ == "__main__":

    memory = MemorySaver()

    agent = build_graph(memory_=memory)
    agent.get_graph().draw_mermaid_png(output_file_path="tmp.png")

    config = {"configurable": {"thread_id": "1"}}

    def stream_graph_updates(user_input_: str):
        for event in agent.stream(
            {"messages": [{"role": "user", "content": user_input_}]},
            config=config,  # type: ignore[arg-type]
        ):
            for value in event.values():
                if "messages" in value:
                    # print("Assistant:", value["messages"][-1].content)
                    value["messages"][-1].pretty_print()
                    print()

    def stream_graph_after_interrupt(user_input_: str):
        for event in agent.stream(
            Command(resume=user_input_),
            config=config,  # type: ignore[arg-type]
        ):
            for value in event.values():
                if "messages" in value:
                    value["messages"][-1].pretty_print()
                    print()

    while True:

        if (
            len(
                agent.get_state(
                    config  # pyright: ignore[reportArgumentType]
                ).interrupts
            )
            > 0
        ):
            print()
            print(
                agent.get_state(config)  # pyright: ignore[reportArgumentType]
                .interrupts[0]
                .value["question"]
            )
            print()
            user_input = input("Y/N: ")
            while user_input not in ["Y", "N"]:
                print("Answer with Y or N please")
                user_input = input("Y/N: ")
            stream_graph_after_interrupt(user_input)

        user_input = input("User: ")

        if user_input == "User: ":
            continue

        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break

        stream_graph_updates(user_input)
