"""
https://langchain-ai.github.io/langgraph/tutorials/chatbots/information-gather-prompting/
"""

import dotenv

dotenv.load_dotenv()

# pylint: disable=C0413
import uuid
from typing import Annotated, List

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph, StateGraph
from pydantic import BaseModel
from typing_extensions import TypedDict


def create_llm_model() -> ChatGoogleGenerativeAI:
    """Create a ChatGoogleGenerativeAI model.

    Returns:
        ChatGoogleGenerativeAI: A ChatGoogleGenerativeAI model.
    """
    return ChatGoogleGenerativeAI(
        # model="gemini-2.0-flash-lite",
        model="gemini-2.0-flash",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
    )


class PromptInstructions(BaseModel):
    """Instructions on how to prompt the LLM."""

    objective: str
    variables: List[str]
    constraints: List[str]
    requirements: List[str]


llm = create_llm_model()
llm_with_tool = llm.bind_tools(
    [PromptInstructions]  # type: ignore[list-item]  # pyright: ignore[reportArgumentType]
)


# -----------------------------------------------------------------------------
# Information gatherer

# pylint: disable=C0301
TEMPLATE = """Your job is to get information from a user about what type of prompt template they want to create.

You should get the following information from them:

- What the objective of the prompt is
- What variables will be passed into the prompt template
- Any constraints for what the output should NOT do
- Any requirements that the output MUST adhere to

If you are not able to discern this info, ask them to clarify! Do not attempt to wildly guess.

After you are able to discern all the information, call the relevant tool."""


def get_messages_info(messages):
    return [SystemMessage(content=TEMPLATE)] + messages


def info_chain(state):
    messages = get_messages_info(state["messages"])
    response = llm_with_tool.invoke(messages)
    return {"messages": [response]}


# -----------------------------------------------------------------------------
# Prompt generator

PROMPT_SYSTEM = """Based on the following requirements, write a good prompt template:

{reqs}"""


# Function to get the messages for the prompt
# Will only get messages AFTER the tool call
def get_prompt_messages(messages: list):
    tool_call = None
    other_msgs = []
    for m in messages:
        if isinstance(m, AIMessage) and m.tool_calls:
            tool_call = m.tool_calls[0]["args"]
        elif isinstance(m, ToolMessage):
            continue
        elif tool_call is not None:
            other_msgs.append(m)
    # Gemini in langchain does seem to support system message.
    # return [
    #     SystemMessage(content=PROMPT_SYSTEM.format(reqs=tool_call))
    # ] + other_msgs
    return [
        HumanMessage(content=PROMPT_SYSTEM.format(reqs=tool_call))
    ] + other_msgs


def prompt_gen_chain(state):
    messages = get_prompt_messages(state["messages"])
    response = llm.invoke(messages)
    return {"messages": [response]}


# -----------------------------------------------------------------------------
# Graph


def get_state(state):
    messages = state["messages"]
    if isinstance(messages[-1], AIMessage) and messages[-1].tool_calls:
        return "add_tool_message"
    elif not isinstance(messages[-1], HumanMessage):
        return END
    return "info"


class State(TypedDict):
    messages: Annotated[list, add_messages]


def create_graph() -> CompiledStateGraph:
    """Create a graph for the information gatherer and prompt generator.

    Returns:
        CompiledStateGraph: A compiled state graph for the agent.
    """
    memory = MemorySaver()
    workflow = StateGraph(State)
    workflow.add_node("info", info_chain)
    workflow.add_node("prompt", prompt_gen_chain)

    @workflow.add_node
    def add_tool_message(state: State):
        return {
            "messages": [
                ToolMessage(
                    content="Prompt generated!",
                    tool_call_id=state["messages"][-1].tool_calls[0]["id"],
                )
            ]
        }

    workflow.add_conditional_edges(
        "info",
        get_state,
        ["add_tool_message", "info", END],
    )
    workflow.add_edge("add_tool_message", "prompt")
    workflow.add_edge("prompt", END)
    workflow.add_edge(START, "info")
    return workflow.compile(checkpointer=memory)


# -----------------------------------------------------------------------------
# MAIN
graph = create_graph()
cached_human_responses = [
    "hi!",
    "rag prompt",
    "1 to answer a question about apple, 2 apple, 3 include color red, 4 must be about fruit",
    "yes",
    "q",
]
cached_response_index = 0
config = {"configurable": {"thread_id": str(uuid.uuid4())}}
while True:
    try:
        user = cached_human_responses[cached_response_index]
        cached_response_index += 1
    except:  # pylint: disable=W0702
        user = input("User (q/Q to quit): ")
    print(f"Input: {user}")
    if user in {"q", "Q"}:
        print("AI: Byebye")
        break
    output = None
    for output in graph.stream(
        {"messages": [HumanMessage(content=user)]},
        config=config,  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]
        stream_mode="updates",
    ):
        last_message = next(iter(output.values()))["messages"][-1]
        last_message.pretty_print()

    if output and "prompt" in output:
        print("Done!")
