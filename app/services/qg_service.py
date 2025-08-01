from typing import Literal, TypedDict # <--- ADD TypedDict HERE
from pathlib import Path
from fastapi import HTTPException

from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS # <--- Use the new import path
from langgraph.graph import StateGraph, END

from ..core.config import settings
from .document_service import get_embeddings_model, VECTOR_STORE_PATH
# vvv REMOVE GraphState FROM THIS IMPORT vvv
from ..models.schemas import MCQs, FillInTheBlanks, Summary

# --- Define GraphState HERE, where it is used ---
class GraphState(TypedDict):
    topic: str
    content_type: Literal["MCQ", "FillInTheBlank", "Summary"]
    documents: list[str]
    final_output: dict

# --- Initialize LLM and Vector Store Components ---
llm = ChatGroq(model="llama3-8b-8192", temperature=0, api_key=settings.GROQ_API_KEY)

try:
    embeddings = get_embeddings_model()
    # Note: allow_dangerous_deserialization is needed for FAISS. Trust your source files.
    vector_store = FAISS.load_local(VECTOR_STORE_PATH, embeddings, allow_dangerous_deserialization=True)
    retriever = vector_store.as_retriever(search_kwargs={'k': 5})
except Exception:
    retriever = None

# --- Specialized Agent Nodes ---
def retrieve_documents(state: GraphState) -> GraphState:
    if retriever is None:
        raise FileNotFoundError("Vector store not found. Please ingest a document first.")
    documents = retriever.invoke(state["topic"])
    return {"documents": [doc.page_content for doc in documents], **state}

def mcq_agent(state: GraphState) -> GraphState:
    prompt = ChatPromptTemplate.from_template(
        """**Task:** Generate 3 multiple-choice questions based on the context. Your response must be a single, raw JSON object conforming to the MCQs schema.
        **Context:** {context}"""
    )
    chain = prompt | llm.with_structured_output(MCQs)
    result = chain.invoke({"context": "\n\n".join(state["documents"])})
    return {"final_output": result.dict()}

def fitb_agent(state: GraphState) -> GraphState:
    prompt = ChatPromptTemplate.from_template(
        """**Task:** Generate 3 high-quality fill-in-the-blank questions based on the context. Your response must be a single, raw JSON object.
        A good question replaces a single key term with '_________'.
        **Example:** "When you have a negative exponent, it means _________."
        **Context:** {context}"""
    )
    chain = prompt | llm.with_structured_output(FillInTheBlanks)
    result = chain.invoke({"context": "\n\n".join(state["documents"])})
    return {"final_output": result.dict()}

def summary_agent(state: GraphState) -> GraphState:
    prompt = ChatPromptTemplate.from_template(
        """**Task:** Generate a concise 2-3 sentence summary of the context. Your response must be a single, raw JSON object.
        **Context:** {context}"""
    )
    chain = prompt | llm.with_structured_output(Summary)
    result = chain.invoke({"context": "\n\n".join(state["documents"])})
    return {"final_output": result.dict()}

# --- Router Logic ---
def route_to_agent(state: GraphState) -> str:
    route_map = {"MCQ": "mcq_agent", "FillInTheBlank": "fitb_agent", "Summary": "summary_agent"}
    return route_map[state['content_type']]

# --- Compile LangGraph Workflow ---
workflow = StateGraph(GraphState)
workflow.add_node("retriever", retrieve_documents)
workflow.add_node("mcq_agent", mcq_agent)
workflow.add_node("fitb_agent", fitb_agent)
workflow.add_node("summary_agent", summary_agent)
workflow.set_entry_point("retriever")
workflow.add_conditional_edges("retriever", route_to_agent, {
    "mcq_agent": "mcq_agent", "fitb_agent": "fitb_agent", "summary_agent": "summary_agent"
})
workflow.add_edge("mcq_agent", END)
workflow.add_edge("fitb_agent", END)
workflow.add_edge("summary_agent", END)
app_graph = workflow.compile()

# --- Main Service Function ---
def run_generation(topic: str, content_type: Literal["MCQ", "FillInTheBlank", "Summary"]):
    """Invokes the graph to generate the specified content."""
    global retriever
    if retriever is None:
        if not Path(VECTOR_STORE_PATH).exists():
             raise FileNotFoundError("Vector store not found. Please ingest a document via the /ingest endpoint.")
        try:
             embeddings = get_embeddings_model()
             vector_store = FAISS.load_local(VECTOR_STORE_PATH, embeddings, allow_dangerous_deserialization=True)
             retriever = vector_store.as_retriever(search_kwargs={'k': 5})
        except Exception as e:
             raise RuntimeError(f"Could not load vector store after ingestion: {e}")

    try:
        initial_state = {"topic": topic, "content_type": content_type}
        final_state = app_graph.invoke(initial_state)
        return final_state.get("final_output")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred during content generation: {e}")