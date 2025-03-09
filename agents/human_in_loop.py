"""
Human in loop systems

Based on: https://langchain-ai.github.io/langgraph/concepts/human_in_the_loop/
"""

# from langgraph.types import interrupt

# def human_node(state: State):
#     value = interrupt(
#         # Any JSON serializable value to surface to the human.
#         # For example, a question or a piece of text or a set of keys in the state
#        {
#           "text_to_revise": state["some_text"]
#        }
#     )
#     # Update the state with the human's input or route the graph based on the input.
#     return {
#         "some_text": value
#     }

# graph = graph_builder.compile(
#     checkpointer=checkpointer # Required for `interrupt` to work
# )

# # Run the graph until the interrupt
# thread_config = {"configurable": {"thread_id": "some_id"}}
# graph.invoke(some_input, config=thread_config)

# # Resume the graph with the human's input
# graph.invoke(Command(resume=value_from_human), config=thread_config)

# =============================================================================
# Approve or Reject
# =============================================================================

# from typing import Literal
# from langgraph.types import interrupt, Command

# def human_approval(state: State) -> Command[Literal["some_node", "another_node"]]:
#     is_approved = interrupt(
#         {
#             "question": "Is this correct?",
#             # Surface the output that should be
#             # reviewed and approved by the human.
#             "llm_output": state["llm_output"]
#         }
#     )

#     if is_approved:
#         return Command(goto="some_node")
#     else:
#         return Command(goto="another_node")

# # Add the node to the graph in an appropriate location
# # and connect it to the relevant nodes.
# graph_builder.add_node("human_approval", human_approval)
# graph = graph_builder.compile(checkpointer=checkpointer)

# # After running the graph and hitting the interrupt, the graph will pause.
# # Resume it with either an approval or rejection.
# thread_config = {"configurable": {"thread_id": "some_id"}}
# graph.invoke(Command(resume=True), config=thread_config)

# =============================================================================
# Review & Edit State
# =============================================================================

# from langgraph.types import interrupt

# def human_editing(state: State):
#     ...
#     result = interrupt(
#         # Interrupt information to surface to the client.
#         # Can be any JSON serializable value.
#         {
#             "task": "Review the output from the LLM and make any necessary edits.",
#             "llm_generated_summary": state["llm_generated_summary"]
#         }
#     )

#     # Update the state with the edited text
#     return {
#         "llm_generated_summary": result["edited_text"]
#     }

# # Add the node to the graph in an appropriate location
# # and connect it to the relevant nodes.
# graph_builder.add_node("human_editing", human_editing)
# graph = graph_builder.compile(checkpointer=checkpointer)

# ...

# # After running the graph and hitting the interrupt, the graph will pause.
# # Resume it with the edited text.
# thread_config = {"configurable": {"thread_id": "some_id"}}
# graph.invoke(
#     Command(resume={"edited_text": "The edited text"}),
#     config=thread_config
# )

# =============================================================================
# Review Tool Calls
# =============================================================================

# def human_review_node(state) -> Command[Literal["call_llm", "run_tool"]]:
#     # This is the value we'll be providing via Command(resume=<human_review>)
#     human_review = interrupt(
#         {
#             "question": "Is this correct?",
#             # Surface tool calls for review
#             "tool_call": tool_call
#         }
#     )

#     review_action, review_data = human_review

#     # Approve the tool call and continue
#     if review_action == "continue":
#         return Command(goto="run_tool")

#     # Modify the tool call manually and then continue
#     elif review_action == "update":
#         ...
#         updated_msg = get_updated_msg(review_data)
#         # Remember that to modify an existing message you will need
#         # to pass the message with a matching ID.
#         return Command(goto="run_tool", update={"messages": [updated_message]})

#     # Give natural language feedback, and then pass that back to the agent
#     elif review_action == "feedback":
#         ...
#         feedback_msg = get_feedback_msg(review_data)
#         return Command(goto="call_llm", update={"messages": [feedback_msg]})

# =============================================================================
# Multi-turn conversation
# =============================================================================

# from langgraph.types import interrupt

# def human_input(state: State):
#     human_message = interrupt("human_input")
#     return {
#         "messages": [
#             {
#                 "role": "human",
#                 "content": human_message
#             }
#         ]
#     }

# def agent(state: State):
#     # Agent logic
#     ...

# graph_builder.add_node("human_input", human_input)
# graph_builder.add_edge("human_input", "agent")
# graph = graph_builder.compile(checkpointer=checkpointer)

# # After running the graph and hitting the interrupt, the graph will pause.
# # Resume it with the human's input.
# graph.invoke(
#     Command(resume="hello!"),
#     config=thread_config
# )


# from langgraph.types import interrupt

# def human_node(state: MessagesState) -> Command[Literal["agent_1", "agent_2", ...]]:
#     """A node for collecting user input."""
#     user_input = interrupt(value="Ready for user input.")

#     # Determine the **active agent** from the state, so
#     # we can route to the correct agent after collecting input.
#     # For example, add a field to the state or use the last active agent.
#     # or fill in `name` attribute of AI messages generated by the agents.
#     active_agent = ...

#     return Command(
#         update={
#             "messages": [{
#                 "role": "human",
#                 "content": user_input,
#             }]
#         },
#         goto=active_agent,
#     )

# =============================================================================
# Validating human input
# =============================================================================

# from langgraph.types import interrupt

# def human_node(state: State):
#     """Human node with validation."""
#     question = "What is your age?"

#     while True:
#         answer = interrupt(question)

#         # Validate answer, if the answer isn't valid ask for input again.
#         if not isinstance(answer, int) or answer < 0:
#             question = f"'{answer} is not a valid age. What is your age?"
#             answer = None
#             continue
#         else:
#             # If the answer is valid, we can proceed.
#             break

#     print(f"The human in the loop is {answer} years old.")
#     return {
#         "age": answer
#     }

# =============================================================================
# How does resuming from an interrupt work?

# A critical aspect of using interrupt is understanding how resuming works.
# When you resume execution after an interrupt, graph execution starts from
# the beginning of the graph node where the last interrupt was triggered.

# All code from the beginning of the node to the interrupt will be re-executed.
# =============================================================================

# counter = 0
# def node(state: State):
#     # All the code from the beginning of the node to the interrupt will be re-executed
#     # when the graph resumes.
#     global counter
#     counter += 1
#     print(f"> Entered the node: {counter} # of times")
#     # Pause the graph and wait for user input.
#     answer = interrupt()
#     print("The value of counter is:", counter)
#     ...

# import uuid
# from typing import TypedDict

# from langgraph.graph import StateGraph
# from langgraph.constants import START
# from langgraph.types import interrupt, Command
# from langgraph.checkpoint.memory import MemorySaver


# class State(TypedDict):
#    """The graph state."""
#    state_counter: int


# counter_node_in_subgraph = 0

# def node_in_subgraph(state: State):
#    """A node in the sub-graph."""
#    global counter_node_in_subgraph
#    counter_node_in_subgraph += 1  # This code will **NOT** run again!
#    print(f"Entered `node_in_subgraph` a total of {counter_node_in_subgraph} times")

# counter_human_node = 0

# def human_node(state: State):
#    global counter_human_node
#    counter_human_node += 1 # This code will run again!
#    print(f"Entered human_node in sub-graph a total of {counter_human_node} times")
#    answer = interrupt("what is your name?")
#    print(f"Got an answer of {answer}")


# checkpointer = MemorySaver()

# subgraph_builder = StateGraph(State)
# subgraph_builder.add_node("some_node", node_in_subgraph)
# subgraph_builder.add_node("human_node", human_node)
# subgraph_builder.add_edge(START, "some_node")
# subgraph_builder.add_edge("some_node", "human_node")
# subgraph = subgraph_builder.compile(checkpointer=checkpointer)


# counter_parent_node = 0

# def parent_node(state: State):
#    """This parent node will invoke the subgraph."""
#    global counter_parent_node

#    counter_parent_node += 1 # This code will run again on resuming!
#    print(f"Entered `parent_node` a total of {counter_parent_node} times")

#    # Please note that we're intentionally incrementing the state counter
#    # in the graph state as well to demonstrate that the subgraph update
#    # of the same key will not conflict with the parent graph (until
#    subgraph_state = subgraph.invoke(state)
#    return subgraph_state


# builder = StateGraph(State)
# builder.add_node("parent_node", parent_node)
# builder.add_edge(START, "parent_node")

# # A checkpointer must be enabled for interrupts to work!
# checkpointer = MemorySaver()
# graph = builder.compile(checkpointer=checkpointer)

# config = {
#    "configurable": {
#       "thread_id": uuid.uuid4(),
#    }
# }

# for chunk in graph.stream({"state_counter": 1}, config):
#    print(chunk)

# print('--- Resuming ---')

# for chunk in graph.stream(Command(resume="35"), config):
#    print(chunk)