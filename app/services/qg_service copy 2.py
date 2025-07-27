from typing import Literal, TypedDict, List, Optional # <--- ADD Optional HERE
from pathlib import Path
from fastapi import HTTPException

from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langgraph.graph import StateGraph, END
from langchain_core.documents import Document

from ..core.config import settings
from .document_service import get_embeddings_model, VECTOR_STORE_PATH
from ..models.schemas import MCQs, FillInTheBlanks, Summary

# --- Graph State Definition (now includes all parameters) ---
class GraphState(TypedDict):
    topic: Optional[str] # <--- Changed here
    content_type: Literal["MCQ", "FillInTheBlank", "Summary"]
    num_questions: int
    context_chunks: int
    documents: List[Document]
    final_output: dict

# --- Initialize LLM ---
llm = ChatGroq(model="llama3-8b-8192", temperature=0, api_key=settings.GROQ_API_KEY)
retriever = None

# --- Agent Nodes (Updated) ---
def retrieve_documents(state: GraphState) -> GraphState:
    print("--- Node: Retrieving documents ---")
    global retriever
    if retriever is None:
        if not Path(VECTOR_STORE_PATH).exists():
             raise FileNotFoundError("Vector store not found. Please ingest a document first.")
        embeddings = get_embeddings_model()
        vector_store = FAISS.load_local(VECTOR_STORE_PATH, embeddings, allow_dangerous_deserialization=True)
        retriever = vector_store.as_retriever()
    
    retriever.search_kwargs['k'] = state['context_chunks']
    
    topic = state.get("topic")
    if topic:
        print(f"Retrieving {state['context_chunks']} chunks for topic: {topic}")
        documents = retriever.invoke(topic)
    else:
        print(f"No topic provided. Retrieving {state['context_chunks']} general chunks.")
        documents = retriever.invoke("general algebra concepts review")[:state['context_chunks']] # Changed generic query for no topic

    return {"documents": documents, **state}

def get_context_with_sources(documents: List[Document]) -> str:
    return "\n\n".join(
        f"Source Page: {doc.metadata.get('page', 'N/A')}\nContent: {doc.page_content}"
        for doc in documents
    )

def mcq_agent(state: GraphState) -> GraphState:
    print("--- Node: MCQ Agent ---")
    context_with_sources = get_context_with_sources(state["documents"])
    prompt = ChatPromptTemplate.from_template(
        """**Task:** Based on the context below, generate {num_questions} high-quality multiple-choice questions.
        - Your response MUST be a single, raw JSON object.
        - Each question MUST include a `source_page` field, referencing the page number from the context.
        - If you cannot generate the requested number of high-quality questions, generate as many as you can and stop. Do not make up answers.

        **Schema:**
        ```
        class MCQ(BaseModel):
            question: str
            options: List[str]
            correct_answer: str
            explanation: str
            source_page: int
        ```

        **Context with Sources:**
        ---
        {context}
        ---
        """
    )
    chain = prompt | llm.with_structured_output(MCQs)
    result = chain.invoke({"context": context_with_sources, "num_questions": state["num_questions"]})
    return {"final_output": result.dict()}

def fitb_agent(state: GraphState) -> GraphState:
    print("--- Node: Fill-in-the-Blank Agent ---")
    context_with_sources = get_context_with_sources(state["documents"])
    prompt = ChatPromptTemplate.from_template(
        """**Task:** Based on the context below, generate {num_questions} high-quality fill-in-the-blank questions.
        - Your response MUST be a single, raw JSON object.
        - Each question MUST include a `source_page` field, referencing the page number from the context.
        - Replace a single key term with '_________'.
        - If you cannot generate the requested number of questions, generate as many as you can and stop.

        **Context with Sources:**
        ---
        {context}
        ---
        """
    )
    chain = prompt | llm.with_structured_output(FillInTheBlanks)
    result = chain.invoke({"context": context_with_sources, "num_questions": state["num_questions"]})
    return {"final_output": result.dict()}

def summary_agent(state: GraphState) -> GraphState:
    print("--- Node: Summary Agent ---")
    context_with_sources = get_context_with_sources(state["documents"])
    source_pages = sorted(list(set(doc.metadata.get("page", 0) for doc in state["documents"])))
    prompt = ChatPromptTemplate.from_template(
        """**Task:** Based on the context below, generate a concise summary.
        - Your response MUST be a single, raw JSON object.
        - The summary MUST include a `source_pages` field listing all the unique page numbers used from the context.

        **Schema:**
        ```
        class Summary(BaseModel):
            summary_text: str
            source_pages: List[int]
        ```

        **Context with Sources:**
        ---
        {context}
        ---
        """
    )
    chain = prompt | llm.with_structured_output(Summary)
    result = chain.invoke({"context": context_with_sources})
    result.source_pages = source_pages
    return {"final_output": result.dict()}

# --- Router and Graph (No changes needed here) ---
def route_to_agent(state: GraphState) -> str:
    route_map = {"MCQ": "mcq_agent", "FillInTheBlank": "fitb_agent", "Summary": "summary_agent"}
    return route_map[state['content_type']]

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

# --- Main Service Function (Updated to pass all params) ---
def run_generation(topic: Optional[str], content_type: Literal["MCQ", "FillInTheBlank", "Summary"], num_questions: int, context_chunks: int): # <--- Changed topic type here
    try:
        initial_state = {
            "topic": topic,
            "content_type": content_type,
            "num_questions": num_questions,
            "context_chunks": context_chunks,
        }
        final_state = app_graph.invoke(initial_state)
        return final_state.get("final_output")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred during content generation: {str(e)}")