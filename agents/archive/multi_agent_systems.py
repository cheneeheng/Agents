"""
Multi-agent systems

Based on: https://langchain-ai.github.io/langgraph/concepts/multi_agent/
"""

# =============================================================================
# Handoffs
# =============================================================================

# def agent(state) -> Command[Literal["agent", "another_agent"]]:
#     # the condition for routing/halting can be anything, e.g. LLM tool call / structured output, etc.
#     goto = get_next_agent(...)  # 'agent' / 'another_agent'
#     return Command(
#         # Specify which agent to call next
#         goto=goto,
#         # Update the graph state
#         update={"my_state_key": "my_state_value"}
#     )

# def some_node_inside_alice(state)
#     return Command(
#         goto="bob",
#         update={"my_state_key": "my_state_value"},
#         # specify which graph to navigate to (defaults to the current graph)
#         graph=Command.PARENT,
#     )


# def transfer_to_bob(state):
#     """Transfer to bob."""
#     return Command(
#         goto="bob",
#         update={"my_state_key": "my_state_value"},
#         graph=Command.PARENT,
#     )

# =============================================================================
# Network
# =============================================================================

# from typing import Literal
# from langchain_openai import ChatOpenAI
# from langgraph.types import Command
# from langgraph.graph import StateGraph, MessagesState, START, END

# model = ChatOpenAI()

# def agent_1(state: MessagesState) -> Command[Literal["agent_2", "agent_3", END]]:
#     # you can pass relevant parts of the state to the LLM (e.g., state["messages"])
#     # to determine which agent to call next. a common pattern is to call the model
#     # with a structured output (e.g. force it to return an output with a "next_agent" field)
#     response = model.invoke(...)
#     # route to one of the agents or exit based on the LLM's decision
#     # if the LLM returns "__end__", the graph will finish execution
#     return Command(
#         goto=response["next_agent"],
#         update={"messages": [response["content"]]},
#     )

# def agent_2(state: MessagesState) -> Command[Literal["agent_1", "agent_3", END]]:
#     response = model.invoke(...)
#     return Command(
#         goto=response["next_agent"],
#         update={"messages": [response["content"]]},
#     )

# def agent_3(state: MessagesState) -> Command[Literal["agent_1", "agent_2", END]]:
#     ...
#     return Command(
#         goto=response["next_agent"],
#         update={"messages": [response["content"]]},
#     )

# builder = StateGraph(MessagesState)
# builder.add_node(agent_1)
# builder.add_node(agent_2)
# builder.add_node(agent_3)

# builder.add_edge(START, "agent_1")
# network = builder.compile()

# =============================================================================
# Supervisor
# =============================================================================

# from typing import Literal
# from langchain_openai import ChatOpenAI
# from langgraph.types import Command
# from langgraph.graph import StateGraph, MessagesState, START, END

# model = ChatOpenAI()

# def supervisor(state: MessagesState) -> Command[Literal["agent_1", "agent_2", END]]:
#     # you can pass relevant parts of the state to the LLM (e.g., state["messages"])
#     # to determine which agent to call next. a common pattern is to call the model
#     # with a structured output (e.g. force it to return an output with a "next_agent" field)
#     response = model.invoke(...)
#     # route to one of the agents or exit based on the supervisor's decision
#     # if the supervisor returns "__end__", the graph will finish execution
#     return Command(goto=response["next_agent"])

# def agent_1(state: MessagesState) -> Command[Literal["supervisor"]]:
#     # you can pass relevant parts of the state to the LLM (e.g., state["messages"])
#     # and add any additional logic (different models, custom prompts, structured output, etc.)
#     response = model.invoke(...)
#     return Command(
#         goto="supervisor",
#         update={"messages": [response]},
#     )

# def agent_2(state: MessagesState) -> Command[Literal["supervisor"]]:
#     response = model.invoke(...)
#     return Command(
#         goto="supervisor",
#         update={"messages": [response]},
#     )

# builder = StateGraph(MessagesState)
# builder.add_node(supervisor)
# builder.add_node(agent_1)
# builder.add_node(agent_2)

# builder.add_edge(START, "supervisor")

# supervisor = builder.compile()

# =============================================================================
# Supervisor (tool-calling)
# =============================================================================

# from typing import Annotated

# from langchain_openai import ChatOpenAI
# from langgraph.prebuilt import InjectedState, create_react_agent

# model = ChatOpenAI()


# # this is the agent function that will be called as tool
# # notice that you can pass the state to the tool via InjectedState annotation
# def agent_1(state: Annotated[dict, InjectedState]):
#     # you can pass relevant parts of the state to the LLM (e.g., state["messages"])
#     # and add any additional logic (different models, custom prompts, structured output, etc.)
#     response = model.invoke(...)
#     # return the LLM response as a string (expected tool response format)
#     # this will be automatically turned to ToolMessage
#     # by the prebuilt create_react_agent (supervisor)
#     return response.content


# def agent_2(state: Annotated[dict, InjectedState]):
#     response = model.invoke(...)
#     return response.content


# tools = [agent_1, agent_2]
# # the simplest way to build a supervisor w/ tool-calling is to use prebuilt ReAct agent graph
# # that consists of a tool-calling LLM node (i.e. supervisor) and a tool-executing node
# supervisor = create_react_agent(model, tools)

# =============================================================================
# Hierarchical
# =============================================================================

# from typing import Literal
# from langchain_openai import ChatOpenAI
# from langgraph.graph import StateGraph, MessagesState, START, END
# from langgraph.types import Command
# model = ChatOpenAI()

# # define team 1 (same as the single supervisor example above)

# def team_1_supervisor(state: MessagesState) -> Command[Literal["team_1_agent_1", "team_1_agent_2", END]]:
#     response = model.invoke(...)
#     return Command(goto=response["next_agent"])

# def team_1_agent_1(state: MessagesState) -> Command[Literal["team_1_supervisor"]]:
#     response = model.invoke(...)
#     return Command(goto="team_1_supervisor", update={"messages": [response]})

# def team_1_agent_2(state: MessagesState) -> Command[Literal["team_1_supervisor"]]:
#     response = model.invoke(...)
#     return Command(goto="team_1_supervisor", update={"messages": [response]})

# team_1_builder = StateGraph(Team1State)
# team_1_builder.add_node(team_1_supervisor)
# team_1_builder.add_node(team_1_agent_1)
# team_1_builder.add_node(team_1_agent_2)
# team_1_builder.add_edge(START, "team_1_supervisor")
# team_1_graph = team_1_builder.compile()

# # define team 2 (same as the single supervisor example above)
# class Team2State(MessagesState):
#     next: Literal["team_2_agent_1", "team_2_agent_2", "__end__"]

# def team_2_supervisor(state: Team2State):
#     ...

# def team_2_agent_1(state: Team2State):
#     ...

# def team_2_agent_2(state: Team2State):
#     ...

# team_2_builder = StateGraph(Team2State)
# ...
# team_2_graph = team_2_builder.compile()


# # define top-level supervisor

# builder = StateGraph(MessagesState)
# def top_level_supervisor(state: MessagesState) -> Command[Literal["team_1_graph", "team_2_graph", END]]:
#     # you can pass relevant parts of the state to the LLM (e.g., state["messages"])
#     # to determine which team to call next. a common pattern is to call the model
#     # with a structured output (e.g. force it to return an output with a "next_team" field)
#     response = model.invoke(...)
#     # route to one of the teams or exit based on the supervisor's decision
#     # if the supervisor returns "__end__", the graph will finish execution
#     return Command(goto=response["next_team"])

# builder = StateGraph(MessagesState)
# builder.add_node(top_level_supervisor)
# builder.add_node("team_1_graph", team_1_graph)
# builder.add_node("team_2_graph", team_2_graph)
# builder.add_edge(START, "top_level_supervisor")
# builder.add_edge("team_1_graph", "top_level_supervisor")
# builder.add_edge("team_2_graph", "top_level_supervisor")
# graph = builder.compile()
