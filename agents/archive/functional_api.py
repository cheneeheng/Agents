"""
Functional API

Based on: https://langchain-ai.github.io/langgraph/concepts/functional_api/
"""

# from langgraph.func import entrypoint, task
# from langgraph.types import interrupt

# @task
# def write_essay(topic: str) -> str:
#     """Write an essay about the given topic."""
#     time.sleep(1) # A placeholder for a long-running task.
#     return f"An essay about topic: {topic}"

# @entrypoint(checkpointer=MemorySaver())
# def workflow(topic: str) -> dict:
#     """A simple workflow that writes an essay and asks for a review."""
#     essay = write_essay("cat").result()
#     is_approved = interrupt({
#         # Any json-serializable payload provided to interrupt as argument.
#         # It will be surfaced on the client side as an Interrupt when streaming data
#         # from the workflow.
#         "essay": essay, # The essay we want reviewed.
#         # We can add any additional information that we need.
#         # For example, introduce a key called "action" with some instructions.
#         "action": "Please approve/reject the essay",
#     })

#     return {
#         "essay": essay, # The essay that was generated
#         "is_approved": is_approved, # Response from HIL
#     }

# =============================================================================
# Parallel execution
# =============================================================================

# @task
# def add_one(number: int) -> int:
#     return number + 1

# @entrypoint(checkpointer=checkpointer)
# def graph(numbers: list[int]) -> list[str]:
#     futures = [add_one(i) for i in numbers]
#     return [f.result() for f in futures]

# =============================================================================
# Calling subgraphs
# =============================================================================

# from langgraph.func import entrypoint
# from langgraph.graph import StateGraph

# builder = StateGraph()
# ...
# some_graph = builder.compile()

# @entrypoint()
# def some_workflow(some_input: dict) -> int:
#     # Call a graph defined using the graph API
#     result_1 = some_graph.invoke(...)
#     # Call another graph defined using the graph API
#     result_2 = another_graph.invoke(...)
#     return {
#         "result_1": result_1,
#         "result_2": result_2
#     }

# =============================================================================
# Calling other entrypoints
# =============================================================================

# @entrypoint() # Will automatically use the checkpointer from the parent entrypoint
# def some_other_workflow(inputs: dict) -> int:
#     return inputs["value"]

# @entrypoint(checkpointer=checkpointer)
# def my_workflow(inputs: dict) -> int:
#     value = some_other_workflow.invoke({"value": 1})
#     return value

# =============================================================================
# Streaming custom data
# =============================================================================

# from langgraph.checkpoint.memory import MemorySaver
# from langgraph.func import entrypoint, task
# from langgraph.types import StreamWriter

# @task
# def add_one(x):
#     return x + 1

# @task
# def add_two(x):
#     return x + 2

# checkpointer = MemorySaver()

# @entrypoint(checkpointer=checkpointer)
# def main(inputs, writer: StreamWriter) -> int:
#     """A simple workflow that adds one and two to a number."""
#     writer("hello") # Write some data to the `custom` stream
#     add_one(inputs['number']).result() # Will write data to the `updates` stream
#     writer("world") # Write some more data to the `custom` stream
#     add_two(inputs['number']).result() # Will write data to the `updates` stream
#     return 5

# config = {
#     "configurable": {
#         "thread_id": "1"
#     }
# }

# for chunk in main.stream({"number": 1}, stream_mode=["custom", "updates"], config=config):
#     print(chunk)

# ('updates', {'add_one': 2})
# ('updates', {'add_two': 3})
# ('custom', 'hello')
# ('custom', 'world')
# ('updates', {'main': 5})

# =============================================================================
# Retry policy
# =============================================================================

# from langgraph.checkpoint.memory import MemorySaver
# from langgraph.func import entrypoint, task
# from langgraph.types import RetryPolicy

# attempts = 0

# # Let's configure the RetryPolicy to retry on ValueError.
# # The default RetryPolicy is optimized for retrying specific network errors.
# retry_policy = RetryPolicy(retry_on=ValueError)

# @task(retry=retry_policy)
# def get_info():
#     global attempts
#     attempts += 1

#     if attempts < 2:
#         raise ValueError('Failure')
#     return "OK"

# checkpointer = MemorySaver()

# @entrypoint(checkpointer=checkpointer)
# def main(inputs, writer):
#     return get_info().result()

# config = {
#     "configurable": {
#         "thread_id": "1"
#     }
# }

# main.invoke({'any_input': 'foobar'}, config=config)


# =============================================================================
# Resuming after an error
# =============================================================================

# import time
# from langgraph.checkpoint.memory import MemorySaver
# from langgraph.func import entrypoint, task
# from langgraph.types import StreamWriter

# # This variable is just used for demonstration purposes to simulate a network failure.
# # It's not something you will have in your actual code.
# attempts = 0

# @task()
# def get_info():
#     """
#     Simulates a task that fails once before succeeding.
#     Raises an exception on the first attempt, then returns "OK" on subsequent tries.
#     """
#     global attempts
#     attempts += 1

#     if attempts < 2:
#         raise ValueError("Failure")  # Simulate a failure on the first attempt
#     return "OK"

# # Initialize an in-memory checkpointer for persistence
# checkpointer = MemorySaver()

# @task
# def slow_task():
#     """
#     Simulates a slow-running task by introducing a 1-second delay.
#     """
#     time.sleep(1)
#     return "Ran slow task."

# @entrypoint(checkpointer=checkpointer)
# def main(inputs, writer: StreamWriter):
#     """
#     Main workflow function that runs the slow_task and get_info tasks sequentially.

#     Parameters:
#     - inputs: Dictionary containing workflow input values.
#     - writer: StreamWriter for streaming custom data.

#     The workflow first executes `slow_task` and then attempts to execute `get_info`,
#     which will fail on the first invocation.
#     """
#     slow_task_result = slow_task().result()  # Blocking call to slow_task
#     get_info().result()  # Exception will be raised here on the first attempt
#     return slow_task_result

# # Workflow execution configuration with a unique thread identifier
# config = {
#     "configurable": {
#         "thread_id": "1"  # Unique identifier to track workflow execution
#     }
# }

# # This invocation will take ~1 second due to the slow_task execution
# try:
#     # First invocation will raise an exception due to the `get_info` task failing
#     main.invoke({'any_input': 'foobar'}, config=config)
# except ValueError:
#     pass  # Handle the failure gracefully
