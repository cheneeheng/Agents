"""
Time Travel 

Based on: https://langchain-ai.github.io/langgraph/concepts/time-travel/    
"""

# =============================================================================
# Replaying
# =============================================================================

# all_checkpoints = []
# for state in graph.get_state_history(thread):
#     all_checkpoints.append(state)

# config = {'configurable': {'thread_id': '1', 'checkpoint_id': 'xyz'}}
# for event in graph.stream(None, config, stream_mode="values"):
#     print(event)

# =============================================================================
# Forking
# =============================================================================

# config = {"configurable": {"thread_id": "1", "checkpoint_id": "xyz"}}
# graph.update_state(config, {"state": "updated state"})

# config = {'configurable': {'thread_id': '1', 'checkpoint_id': 'xyz-fork'}}
# for event in graph.stream(None, config, stream_mode="values"):
#     print(event)
