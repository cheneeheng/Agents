"""
https://langchain-ai.github.io/langgraph/tutorials/rag/langgraph_agentic_rag/
"""

# import getpass
# import os


# def _set_env(key: str):
#     if key not in os.environ:
#         os.environ[key] = getpass.getpass(f"{key}:")


# _set_env("OPENAI_API_KEY")

import pprint
from typing import Annotated, Literal, Sequence

import dotenv
from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.tools.retriever import create_retriever_tool
from langchain_core.tools.simple import Tool
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, START
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

dotenv.load_dotenv()


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


# -----------------------------------------------------------------------------
def create_retriever() -> VectorStoreRetriever:
    """Create a retriever for the Lilian Weng blog posts on LLM agents,
    prompt engineering, and adversarial attacks on LLMs.

    Returns:
        VectorStoreRetriever: A retriever for the Lilian Weng blog posts.
    """
    urls = [
        "https://lilianweng.github.io/posts/2023-06-23-agent/",
        "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
        "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
    ]
    docs = [WebBaseLoader(url).load() for url in urls]
    docs_list = [item for sublist in docs for item in sublist]
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=100, chunk_overlap=50
    )
    doc_splits = text_splitter.split_documents(docs_list)
    # Add to vectorDB
    vectorstore = Chroma.from_documents(
        documents=doc_splits,
        collection_name="rag-chroma",
        embedding=GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004"
        ),
    )
    return vectorstore.as_retriever()


# -----------------------------------------------------------------------------
def create_tools() -> list[Tool]:
    """Create tools for the Lilian Weng blog posts on LLM agents,
    prompt engineering, and adversarial attacks on LLMs.

    Returns:
        list: A list of tools for the Lilian Weng blog posts.
    """
    retriever = create_retriever()
    retriever_tool = create_retriever_tool(
        retriever,
        "retrieve_blog_posts",
        "Search and return information about Lilian Weng blog posts on LLM"
        "agents, prompt engineering, and adversarial attacks on LLMs.",
    )
    return [retriever_tool]


tools = create_tools()


# -----------------------------------------------------------------------------
class AgentState(TypedDict):
    # The add_messages function defines how an update should be processed
    # Default is to replace. add_messages says "append"
    messages: Annotated[Sequence[BaseMessage], add_messages]


# -----------------------------------------------------------------------------
# Edges


def grade_documents(state: AgentState) -> Literal["generate", "rewrite"]:
    """
    Determines whether the retrieved documents are relevant to the question.

    Args:
        state (messages): The current state

    Returns:
        str: A decision for whether the documents are relevant or not
    """
    print("---CHECK RELEVANCE---")

    messages = state["messages"]
    last_message = messages[-1]
    question = messages[0].content
    docs = last_message.content

    # Data model
    class Grade(BaseModel):
        """Binary score for relevance check."""

        binary_score: str = Field(description="Relevance score 'yes' or 'no'")

    # LLM
    model = create_llm_model()
    # LLM with tool and validation
    llm_with_tool = model.with_structured_output(Grade)
    # Prompt
    # pylint: disable=C0301
    prompt = PromptTemplate(
        template="""You are a grader assessing relevance of a retrieved document to a user question. \n
        Here is the retrieved document: \n\n {context} \n\n
        Here is the user question: {question} \n
        If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.""",
        input_variables=["context", "question"],
    )
    # Chain
    chain = prompt | llm_with_tool
    # Run
    scored_result = chain.invoke({"question": question, "context": docs})

    score = (
        scored_result.binary_score  # type: ignore[union-attr]  # pyright: ignore[reportAttributeAccessIssue]
    )

    if score == "yes":
        print("---DECISION: DOCS RELEVANT---")
        return "generate"

    else:
        print("---DECISION: DOCS NOT RELEVANT---")
        print(score)
        return "rewrite"


# -----------------------------------------------------------------------------
# Nodes


def agent(state: AgentState) -> dict[str, list[BaseMessage]]:
    """
    Invokes the agent model to generate a response based on the current
    state. Given the question, it will decide to retrieve using the
    retriever tool, or simply end.

    Args:
        state (messages): The current state

    Returns:
        dict: The updated state with the agent response appended to messages
    """
    print("---CALL AGENT---")
    messages = state["messages"]
    model = create_llm_model()
    model = model.bind_tools(  # type: ignore[union-attr, assignment]
        tools  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]
    )
    response = model.invoke(messages)
    # We return a list, because this will get added to the existing list
    return {"messages": [response]}


def rewrite(state) -> dict[str, list[BaseMessage]]:
    """
    Transform the query to produce a better question.

    Args:
        state (messages): The current state

    Returns:
        dict: The updated state with re-phrased question
    """
    print("---TRANSFORM QUERY---")
    messages = state["messages"]
    question = messages[0].content
    msg = [
        HumanMessage(
            content=f""" \n
    Look at the input and try to reason about the underlying semantic intent / meaning. \n
    Here is the initial question:
    \n ------- \n
    {question}
    \n ------- \n
    Formulate an improved question: """,
        )
    ]
    # Grader
    model = create_llm_model()
    response = model.invoke(msg)
    return {"messages": [response]}


def generate(state: AgentState) -> dict[str, list[BaseMessage]]:
    """
    Generate answer

    Args:
        state (messages): The current state

    Returns:
         dict: The updated state with re-phrased question
    """
    print("---GENERATE---")
    messages = state["messages"]
    question = messages[0].content
    last_message = messages[-1]
    docs = last_message.content
    # Prompt
    prompt = hub.pull("rlm/rag-prompt")
    # LLM
    llm = create_llm_model()
    # # Post-processing
    # def format_docs(docs):
    #     return "\n\n".join(doc.page_content for doc in docs)
    # Chain
    rag_chain = prompt | llm | StrOutputParser()
    # Run
    response = rag_chain.invoke({"context": docs, "question": question})
    return {"messages": [response]}


# print("*" * 20 + "Prompt[rlm/rag-prompt]" + "*" * 20)
# hub.pull("rlm/rag-prompt").pretty_print()  # Show what the prompt looks like


# -----------------------------------------------------------------------------
# Graph


def create_graph() -> CompiledStateGraph:
    """Create a graph for the agent to decide whether to retrieve documents.

    Returns:
        CompiledStateGraph: A compiled state graph for the agent.
    """
    # Define a new graph
    workflow = StateGraph(AgentState)
    # Define the nodes we will cycle between
    workflow.add_node("agent", agent)  # agent
    retrieve = ToolNode(tools)
    workflow.add_node("retrieve", retrieve)  # retrieval
    workflow.add_node("rewrite", rewrite)  # Re-writing the question
    # Generating a response after we know the documents are relevant
    workflow.add_node("generate", generate)
    # Call agent node to decide to retrieve or not
    workflow.add_edge(START, "agent")
    # Decide whether to retrieve
    workflow.add_conditional_edges(
        "agent",
        # Assess agent decision
        tools_condition,
        {
            # Translate the condition outputs to nodes in our graph
            "tools": "retrieve",
            END: END,
        },
    )
    # Edges taken after the `action` node is called.
    workflow.add_conditional_edges(
        "retrieve",
        # Assess agent decision
        grade_documents,
    )
    workflow.add_edge("generate", END)
    workflow.add_edge("rewrite", "agent")
    # Compile
    return workflow.compile()


# -----------------------------------------------------------------------------
# MAIN
inputs = {
    "messages": [
        ("user", "What does Lilian Weng say about the types of agent memory?"),
    ]
}
graph = create_graph()
for output in graph.stream(inputs):
    for key, value in output.items():
        pprint.pprint(f"Output from node '{key}':")
        pprint.pprint("---")
        pprint.pprint(value, indent=2, width=80, depth=None)
    pprint.pprint("\n---\n")
