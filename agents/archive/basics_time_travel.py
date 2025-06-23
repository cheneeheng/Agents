"""
Basics of using langgraph. Part6: Time Travel

Based on: https://langchain-ai.github.io/langgraph/tutorials/introduction/
"""

from typing import Annotated

import dotenv
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import AnyMessage
from langchain_core.runnables import RunnableConfig
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import StateSnapshot
from typing_extensions import TypedDict

dotenv.load_dotenv()


class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


graph_builder = StateGraph(State)


tool = TavilySearchResults(max_results=2)
tools = [tool]
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)
llm_with_tools = llm.bind_tools(tools)  # type: ignore[arg-type]


def chatbot(state_: State):
    return {"messages": [llm_with_tools.invoke(state_["messages"])]}


graph_builder.add_node("chatbot", chatbot)

tool_node = ToolNode(tools=[tool])
graph_builder.add_node("tools", tool_node)

graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
)
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge(START, "chatbot")

memory = MemorySaver()
graph = graph_builder.compile(checkpointer=memory)

config = RunnableConfig({"configurable": {"thread_id": "1"}})

# Conversation 1
events = graph.stream(
    {
        "messages": [
            {
                "role": "user",
                "content": (
                    "I'm learning LangGraph. "
                    "Could you do some research on it for me?"
                ),
            },
        ],
    },
    config,
    stream_mode="values",
)
for event in events:
    if "messages" in event:
        event["messages"][-1].pretty_print()

# Conversation 2
events = graph.stream(
    {
        "messages": [
            {
                "role": "user",
                "content": (
                    "Ya that's helpful. Maybe I'll "
                    "build an autonomous agent with it! "
                    "Can you search for some tutorials?"
                ),
            },
        ],
    },
    config,
    stream_mode="values",
)
for event in events:
    if "messages" in event:
        event["messages"][-1].pretty_print()

# Replay the conversation
to_replay: StateSnapshot | None = None
for state in graph.get_state_history(config):
    print("Num Messages: ", len(state.values["messages"]), "Next: ", state.next)
    print("-" * 80)
    if len(state.values["messages"]) == 6:
        # We are somewhat arbitrarily selecting a specific state
        # based on the number of chat messages in the state.
        to_replay = state

if to_replay is not None:
    print(to_replay.next)
    print(to_replay.config)

if to_replay is not None:
    # The `checkpoint_id` in the `to_replay.config` corresponds
    # to a state we've persisted to our checkpointer.
    for event in graph.stream(
        None,
        to_replay.config,
        stream_mode="values",
    ):
        if "messages" in event:
            event["messages"][-1].pretty_print()
