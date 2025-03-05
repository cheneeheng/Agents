"""
LangGraph glossary

Based on: https://langchain-ai.github.io/langgraph/concepts/low_level/
"""

# =============================================================================
# State
# =============================================================================

# class InputState(TypedDict):
#     user_input: str

# class OutputState(TypedDict):
#     graph_output: str

# class OverallState(TypedDict):
#     foo: str
#     user_input: str
#     graph_output: str

# class PrivateState(TypedDict):
#     bar: str

# def node_1(state: InputState) -> OverallState:
#     # Write to OverallState
#     return {"foo": state["user_input"] + " name"}

# def node_2(state: OverallState) -> PrivateState:
#     # Read from OverallState, write to PrivateState
#     return {"bar": state["foo"] + " is"}

# def node_3(state: PrivateState) -> OutputState:
#     # Read from PrivateState, write to OutputState
#     return {"graph_output": state["bar"] + " Lance"}

# builder = StateGraph(OverallState,input=InputState,output=OutputState)
# builder.add_node("node_1", node_1)
# builder.add_node("node_2", node_2)
# builder.add_node("node_3", node_3)
# builder.add_edge(START, "node_1")
# builder.add_edge("node_1", "node_2")
# builder.add_edge("node_2", "node_3")
# builder.add_edge("node_3", END)

# graph = builder.compile()
# graph.invoke({"user_input":"My"})
# {'graph_output': 'My name is Lance'}

# =============================================================================
# Reducer
# =============================================================================

# from typing_extensions import TypedDict

# class State(TypedDict):
#     foo: int
#     bar: list[str]

# from typing import Annotated
# from typing_extensions import TypedDict
# from operator import add

# class State(TypedDict):
#     foo: int
#     bar: Annotated[list[str], add]

# =============================================================================
# Messages
# =============================================================================

# # this is supported
# {"messages": [HumanMessage(content="message")]}

# # and this is also supported
# {"messages": [{"type": "human", "content": "message"}]}

# from langchain_core.messages import AnyMessage
# from langgraph.graph.message import add_messages
# from typing import Annotated
# from typing_extensions import TypedDict

# class GraphState(TypedDict):
#     messages: Annotated[list[AnyMessage], add_messages]

# from langgraph.graph import MessagesState

# class State(MessagesState):
#     documents: list[str]

# =============================================================================
# Nodes
# =============================================================================

# from langchain_core.runnables import RunnableConfig
# from langgraph.graph import StateGraph

# builder = StateGraph(dict)


# def my_node(state: dict, config: RunnableConfig):
#     print("In node: ", config["configurable"]["user_id"])
#     return {"results": f"Hello, {state['input']}!"}


# # The second argument is optional
# def my_other_node(state: dict):
#     return state


# builder.add_node("my_node", my_node)
# builder.add_node("other_node", my_other_node)
# ...

# builder.add_node(my_node)
# # You can then create edges to/from this node by referencing it as `"my_node"`

# from langgraph.graph import START

# graph.add_edge(START, "node_a")

# from langgraph.graph import END

# graph.add_edge("node_a", END)

# =============================================================================
# Edges
# =============================================================================

# graph.add_edge("node_a", "node_b")

# graph.add_conditional_edges("node_a", routing_function)

# graph.add_conditional_edges("node_a", routing_function, {True: "node_b", False: "node_c"})

# from langgraph.graph import START

# graph.add_edge(START, "node_a")

# from langgraph.graph import START

# graph.add_conditional_edges(START, routing_function)

# graph.add_conditional_edges(START, routing_function, {True: "node_b", False: "node_c"})


# =============================================================================
# Send
# =============================================================================

# def continue_to_jokes(state: OverallState):
#     return [Send("generate_joke", {"subject": s}) for s in state['subjects']]

# graph.add_conditional_edges("node_a", continue_to_jokes)


# =============================================================================
# Command
# =============================================================================

# def my_node(state: State) -> Command[Literal["my_other_node"]]:
#     return Command(
#         # state update
#         update={"foo": "bar"},
#         # control flow
#         goto="my_other_node"
#     )

# def my_node(state: State) -> Command[Literal["my_other_node"]]:
#     if state["foo"] == "bar":
#         return Command(update={"foo": "baz"}, goto="my_other_node")

# def my_node(state: State) -> Command[Literal["my_other_node"]]:
#     return Command(
#         update={"foo": "bar"},
#         goto="other_subgraph",  # where `other_subgraph` is a node in the parent graph
#         graph=Command.PARENT
#     )

# @tool
# def lookup_user_info(tool_call_id: Annotated[str, InjectedToolCallId], config: RunnableConfig):
#     """Use this to look up user information to better assist them with their questions."""
#     user_info = get_user_info(config.get("configurable", {}).get("user_id"))
#     return Command(
#         update={
#             # update the state keys
#             "user_info": user_info,
#             # update the message history
#             "messages": [ToolMessage("Successfully looked up user information", tool_call_id=tool_call_id)]
#         }
#     )
