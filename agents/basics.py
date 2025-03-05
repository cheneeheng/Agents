"""
Basics of using langgraph.

Based on: https://langchain-ai.github.io/langgraph/tutorials/introduction/
"""

import json
from pprint import pprint
from typing import Annotated, Any, Callable, cast

import dotenv
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import AIMessage, AnyMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import InjectedToolCallId, tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph  # pylint: disable=W0611
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import Command, interrupt
from typing_extensions import TypedDict

dotenv.load_dotenv()


class BasicToolNode:
    """
    A node that runs the tools requested in the last AIMessage.

    Same as: langgraph.prebuilt.toolnode.ToolNode
    """

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


def route_tools(
    state: list[AIMessage] | dict[str, list[AIMessage]]
) -> str:  # pylint: disable=W0622
    """
    Use in the conditional_edge to route to the ToolNode if the last message
    has tool calls. Otherwise, route to the end.

    Same as: langgraph.prebuilt.toolnode.tools_condition
    """
    if isinstance(state, list):
        ai_message = state[-1]
    elif messages := state.get("messages", []):
        ai_message = messages[-1]
    else:
        raise ValueError(
            f"No messages found in input state to route_tools: {state}"
        )
    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        return "tools"
    return END


# @tool
# def human_assistance(query: str) -> str:
#     """Request assistance from a human."""
#     human_response = interrupt({"query": query})
#     return human_response["data"]


@tool
# Note that because we are generating a ToolMessage for a state update, we
# generally require the ID of the corresponding tool call. We can use
# LangChain's InjectedToolCallId to signal that this argument should not
# be revealed to the model in the tool's schema.
def human_assistance(
    name: str,
    birthday: str,
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> str:
    """Request assistance from a human."""
    human_response = interrupt(
        {
            "question": "Is this correct?",
            "name": name,
            "birthday": birthday,
        },
    )
    # If the information is correct, update the state as-is.
    if human_response.get("correct", "").lower().startswith("y"):
        verified_name = name
        verified_birthday = birthday
        response = "Correct"
    # Otherwise, receive information from the human reviewer.
    else:
        verified_name = human_response.get("name", name)
        verified_birthday = human_response.get("birthday", birthday)
        response = f"Made a correction: {human_response}"

    # This time we explicitly update the state with a ToolMessage inside
    # the tool.
    state_update = {
        "name": verified_name,
        "birthday": verified_birthday,
        "messages": [ToolMessage(response, tool_call_id=tool_call_id)],
    }
    # We return a Command object in the tool to update our state.
    return Command(update=state_update)  # pyright: ignore [reportReturnType]


class GeminiAgent:
    """Simple gemini agent that uses a chatbot and tools."""

    class State(TypedDict):
        messages: Annotated[list[AnyMessage], add_messages]
        name: str
        birthday: str

    def __init__(self):
        self.tools = [TavilySearchResults(max_results=2), human_assistance]
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
        )
        self.llm_with_tools = self.llm.bind_tools(self.tools)
        self.memory = MemorySaver()
        self.graph = self._build_graph(self.chatbot, self.tools, self.memory)
        # For MemorySaver
        self.config = RunnableConfig(configurable={"thread_id": "1"})

    @staticmethod
    def _build_graph(
        chatbot: Callable[[State], dict[str, list[AIMessage]]],
        tools: list[Any],
        memory: MemorySaver,
    ) -> CompiledStateGraph:
        """Build the state graph for the agent.

        Args:
            chatbot (Callable[[State], dict[str, list[AIMessage]]]):
                Chatbot function.
            tools (list[Any]): List of tools.
            memory (MemorySaver): Memory saver.

        Returns:
            CompiledStateGraph: The compiled state graph.
        """
        graph_builder = StateGraph(GeminiAgent.State)
        graph_builder.add_node("chatbot", chatbot)
        graph_builder.add_node("tools", ToolNode(tools))
        graph_builder.add_conditional_edges("chatbot", tools_condition)
        # Any time a tool is called, we return to the chatbot to decide
        # the next step
        graph_builder.add_edge("tools", "chatbot")
        # The next 2 lines are identical. The second one is just a shorthand.
        # graph_builder.add_edge(START, "chatbot")
        graph_builder.set_entry_point("chatbot")
        graph = graph_builder.compile(checkpointer=memory)
        graph.get_graph().draw_mermaid_png(output_file_path="graph.png")
        return graph

    def chatbot(self, state: State) -> dict[str, list[AIMessage]]:
        message = cast(AIMessage, self.llm_with_tools.invoke(state["messages"]))
        # Because we will be interrupting during tool execution,
        # we disable parallel tool calling to avoid repeating any
        # tool invocations when we resume.
        assert len(message.tool_calls) <= 1
        return {"messages": [message]}

    def get_snapshot(self):
        snapshot = self.graph.get_state(self.config)
        print("=" * 80)
        print("Snapshot: ", snapshot)
        print("Next Snapshot: ", snapshot.next)

    def stream(self, input: str):  # pylint: disable=W0622
        """Stream the input through the graph and print the output.

        Args:
            input (str): The input to the agent.
        """
        # for event in self.graph.stream(
        #     {"messages": [{"role": "user", "content": input}]},
        #     config=self.config,
        # ):
        #     for value in event.values():
        #         print("Assistant:", value["messages"][-1].content)

        events = self.graph.stream(
            {"messages": [{"role": "user", "content": input}]},
            config=self.config,
            stream_mode="values",
        )
        for event in events:
            event["messages"][-1].pretty_print()

        # # For `human_assistance(query: str)` tool
        # if self.graph.get_state(self.config).tasks[0].interrupts != ():
        #     human_response = (
        #         "We, the experts are here to help! We'd recommend you "
        #         "check out LangGraph to build your agent."
        #         " It's much more reliable and extensible than simple "
        #         "autonomous agents."
        #     )
        #     human_command: Command = Command(resume={"data": human_response})
        #     resumed_events = self.graph.stream(
        #         human_command,
        #         config=self.config,
        #         stream_mode="values",
        #     )
        #     for resumed_event in resumed_events:
        #         resumed_event["messages"][-1].pretty_print()

        if self.graph.get_state(self.config).tasks[0].interrupts != ():
            human_command: Command = Command(
                resume={
                    "name": "LangGraph",
                    "birthday": "Jan 17, 2024",
                },
            )
            resumed_events = self.graph.stream(
                human_command,
                config=self.config,
                stream_mode="values",
            )
            for resumed_event in resumed_events:
                resumed_event["messages"][-1].pretty_print()

        # To get the state.
        snapshot = self.graph.get_state(self.config)
        pprint(snapshot.values)

        # To update the state.
        # self.graph.update_state(self.config, {"name": "LangGraph (library)"})


if __name__ == "__main__":

    agent = GeminiAgent()

    # while True:
    #     try:
    #         user_input = input("User: ")
    #         if user_input.lower() in ["quit", "exit", "q"]:
    #             print("Goodbye!")
    #             break

    #         agent.stream(user_input)
    #     except Exception as e:  # pylint: disable=W0702
    #         # fallback if input() is not available
    #         user_input = "What do you know about LangGraph?"
    #         print("User: " + user_input)
    #         agent.stream(user_input)
    #         break

    # user_input = (
    #     "I need some expert guidance for building an AI agent. "
    #     "Could you request assistance for me?"
    # )
    user_input = (
        "Can you look up when LangGraph was released? "
        "When you have the answer, use the human_assistance tool for review."
    )
    agent.stream(user_input)
