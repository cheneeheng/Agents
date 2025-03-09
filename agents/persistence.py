"""
Persistence + Memory

Persistence: LangGraph has a built-in persistence layer, implemented through
checkpointers. This persistence layer helps to support powerful capabilities
like human-in-the-loop, memory, time travel, and fault-tolerance.

Memory: Memory in AI applications refers to the ability to process, store,
and effectively recall information from past interactions. With memory,
your agents can learn from feedback and adapt to users' preferences.

Based on:
https://langchain-ai.github.io/langgraph/concepts/persistence/
https://langchain-ai.github.io/langgraph/concepts/memory/
"""

# =============================================================================
# Threads
# =============================================================================

# {"configurable": {"thread_id": "1"}}

# =============================================================================
# Checkpoints
# =============================================================================

# from langgraph.graph import StateGraph, START, END
# from langgraph.checkpoint.memory import MemorySaver
# from typing import Annotated
# from typing_extensions import TypedDict
# from operator import add

# class State(TypedDict):
#     foo: int
#     bar: Annotated[list[str], add]

# def node_a(state: State):
#     return {"foo": "a", "bar": ["a"]}

# def node_b(state: State):
#     return {"foo": "b", "bar": ["b"]}


# workflow = StateGraph(State)
# workflow.add_node(node_a)
# workflow.add_node(node_b)
# workflow.add_edge(START, "node_a")
# workflow.add_edge("node_a", "node_b")
# workflow.add_edge("node_b", END)

# checkpointer = MemorySaver()
# graph = workflow.compile(checkpointer=checkpointer)

# config = {"configurable": {"thread_id": "1"}}
# graph.invoke({"foo": ""}, config)


# # get the latest state snapshot
# config = {"configurable": {"thread_id": "1"}}
# graph.get_state(config)

# # get a state snapshot for a specific checkpoint_id
# config = {"configurable": {"thread_id": "1", "checkpoint_id": "1ef663ba-28fe-6528-8002-5a559208592c"}}
# graph.get_state(config)

# # Output looks like this:
# StateSnapshot(
#     values={'foo': 'b', 'bar': ['a', 'b']},
#     next=(),
#     config={'configurable': {'thread_id': '1', 'checkpoint_ns': '', 'checkpoint_id': '1ef663ba-28fe-6528-8002-5a559208592c'}},
#     metadata={'source': 'loop', 'writes': {'node_b': {'foo': 'b', 'bar': ['b']}}, 'step': 2},
#     created_at='2024-08-29T19:19:38.821749+00:00',
#     parent_config={'configurable': {'thread_id': '1', 'checkpoint_ns': '', 'checkpoint_id': '1ef663ba-28f9-6ec4-8001-31981c2c39f8'}}, tasks=()
# )

# config = {"configurable": {"thread_id": "1"}}
# list(graph.get_state_history(config))

# # Output looks like this:
# [
#     StateSnapshot(
#         values={'foo': 'b', 'bar': ['a', 'b']},
#         next=(),
#         config={'configurable': {'thread_id': '1', 'checkpoint_ns': '', 'checkpoint_id': '1ef663ba-28fe-6528-8002-5a559208592c'}},
#         metadata={'source': 'loop', 'writes': {'node_b': {'foo': 'b', 'bar': ['b']}}, 'step': 2},
#         created_at='2024-08-29T19:19:38.821749+00:00',
#         parent_config={'configurable': {'thread_id': '1', 'checkpoint_ns': '', 'checkpoint_id': '1ef663ba-28f9-6ec4-8001-31981c2c39f8'}},
#         tasks=(),
#     ),
#     StateSnapshot(
#         values={'foo': 'a', 'bar': ['a']}, next=('node_b',),
#         config={'configurable': {'thread_id': '1', 'checkpoint_ns': '', 'checkpoint_id': '1ef663ba-28f9-6ec4-8001-31981c2c39f8'}},
#         metadata={'source': 'loop', 'writes': {'node_a': {'foo': 'a', 'bar': ['a']}}, 'step': 1},
#         created_at='2024-08-29T19:19:38.819946+00:00',
#         parent_config={'configurable': {'thread_id': '1', 'checkpoint_ns': '', 'checkpoint_id': '1ef663ba-28f4-6b4a-8000-ca575a13d36a'}},
#         tasks=(PregelTask(id='6fb7314f-f114-5413-a1f3-d37dfe98ff44', name='node_b', error=None, interrupts=()),),
#     ),
#     StateSnapshot(
#         values={'foo': '', 'bar': []},
#         next=('node_a',),
#         config={'configurable': {'thread_id': '1', 'checkpoint_ns': '', 'checkpoint_id': '1ef663ba-28f4-6b4a-8000-ca575a13d36a'}},
#         metadata={'source': 'loop', 'writes': None, 'step': 0},
#         created_at='2024-08-29T19:19:38.817813+00:00',
#         parent_config={'configurable': {'thread_id': '1', 'checkpoint_ns': '', 'checkpoint_id': '1ef663ba-28f0-6c66-bfff-6723431e8481'}},
#         tasks=(PregelTask(id='f1b14528-5ee5-579c-949b-23ef9bfbed58', name='node_a', error=None, interrupts=()),),
#     ),
#     StateSnapshot(
#         values={'bar': []},
#         next=('__start__',),
#         config={'configurable': {'thread_id': '1', 'checkpoint_ns': '', 'checkpoint_id': '1ef663ba-28f0-6c66-bfff-6723431e8481'}},
#         metadata={'source': 'input', 'writes': {'foo': ''}, 'step': -1},
#         created_at='2024-08-29T19:19:38.816205+00:00',
#         parent_config=None,
#         tasks=(PregelTask(id='6d27aa2e-d72b-5504-a36f-8620e54a76dd', name='__start__', error=None, interrupts=()),),
#     )
# ]

# # Replay
# config = {"configurable": {"thread_id": "1", "checkpoint_id": "0c62ca34-ac19-445d-bbb0-5b4984975b2a"}}
# graph.invoke(None, config=config)

# # Update state
# from typing import Annotated
# from typing_extensions import TypedDict
# from operator import add

# class State(TypedDict):
#     foo: int
#     bar: Annotated[list[str], add]

# graph.update_state(config, {"foo": 2, "bar": ["b"]})

# =============================================================================
# Memory Store
# =============================================================================

# from langgraph.store.memory import InMemoryStore
# in_memory_store = InMemoryStore()

# user_id = "1"
# namespace_for_memory = (user_id, "memories")

# memory_id = str(uuid.uuid4())
# memory = {"food_preference" : "I like pizza"}
# in_memory_store.put(namespace_for_memory, memory_id, memory)

# memories = in_memory_store.search(namespace_for_memory)
# memories[-1].dict()
# {'value': {'food_preference': 'I like pizza'},
#  'key': '07e0caf4-1631-47b7-b15f-65515d4c1843',
#  'namespace': ['1', 'memories'],
#  'created_at': '2024-10-02T17:22:31.590602+00:00',
#  'updated_at': '2024-10-02T17:22:31.590605+00:00'}

# from langchain.embeddings import init_embeddings

# store = InMemoryStore(
#     index={
#         "embed": init_embeddings("openai:text-embedding-3-small"),  # Embedding provider
#         "dims": 1536,                              # Embedding dimensions
#         "fields": ["food_preference", "$"]              # Fields to embed
#     }
# )

# # Find memories about food preferences
# # (This can be done after putting memories into the store)
# memories = store.search(
#     namespace_for_memory,
#     query="What does the user like to eat?",
#     limit=3  # Return top 3 matches
# )

# # Store with specific fields to embed
# store.put(
#     namespace_for_memory,
#     str(uuid.uuid4()),
#     {
#         "food_preference": "I love Italian cuisine",
#         "context": "Discussing dinner plans"
#     },
#     index=["food_preference"]  # Only embed "food_preferences" field
# )

# # Store without embedding (still retrievable, but not searchable)
# store.put(
#     namespace_for_memory,
#     str(uuid.uuid4()),
#     {"system_info": "Last updated: 2024-01-01"},
#     index=False
# )


# from langgraph.checkpoint.memory import MemorySaver

# # We need this because we want to enable threads (conversations)
# checkpointer = MemorySaver()

# # ... Define the graph ...

# # Compile the graph with the checkpointer and store
# graph = graph.compile(checkpointer=checkpointer, store=in_memory_store)

# # Invoke the graph
# user_id = "1"
# config = {"configurable": {"thread_id": "1", "user_id": user_id}}

# # First let's just say hi to the AI
# for update in graph.stream(
#     {"messages": [{"role": "user", "content": "hi"}]}, config, stream_mode="updates"
# ):
#     print(update)

# def update_memory(state: MessagesState, config: RunnableConfig, *, store: BaseStore):

#     # Get the user id from the config
#     user_id = config["configurable"]["user_id"]

#     # Namespace the memory
#     namespace = (user_id, "memories")

#     # ... Analyze conversation and create a new memory

#     # Create a new memory ID
#     memory_id = str(uuid.uuid4())

#     # We create a new memory
#     store.put(namespace, memory_id, {"memory": memory})

# memories[-1].dict()
# {'value': {'food_preference': 'I like pizza'},
#  'key': '07e0caf4-1631-47b7-b15f-65515d4c1843',
#  'namespace': ['1', 'memories'],
#  'created_at': '2024-10-02T17:22:31.590602+00:00',
#  'updated_at': '2024-10-02T17:22:31.590605+00:00'}

# def call_model(state: MessagesState, config: RunnableConfig, *, store: BaseStore):
#     # Get the user id from the config
#     user_id = config["configurable"]["user_id"]

#     # Search based on the most recent message
#     memories = store.search(
#         namespace,
#         query=state["messages"][-1].content,
#         limit=3
#     )
#     info = "\n".join([d.value["memory"] for d in memories])

#     # ... Use memories in the model call
