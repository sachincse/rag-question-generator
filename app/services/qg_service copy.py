from typing import Literal, TypedDict, List, Optional
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

# --- Graph State Definition ---
class GraphState(TypedDict):
    topic: Optional[str]
    content_type: Literal["MCQ", "FillInTheBlank", "Summary"]
    num_questions: Optional[int]
    context_chunks: int
    documents: List[Document]
    final_output: dict

# --- Initialize LLM ---
llm = ChatGroq(model="llama3-8b-8192", temperature=0, api_key=settings.GROQ_API_KEY)
retriever = None

# --- Agent Nodes ---
def retrieve_documents(state: GraphState) -> GraphState:
    global retriever
    if retriever is None:
        if not Path(VECTOR_STORE_PATH).exists():
            raise FileNotFoundError("Vector store not found. Please ingest a document first.")
        embeddings = get_embeddings_model()
        vector_store = FAISS.load_local(VECTOR_STORE_PATH, embeddings, allow_dangerous_deserialization=True)
        retriever = vector_store.as_retriever()
    
    retriever.search_kwargs['k'] = state['context_chunks']
    topic = state.get("topic")
    documents = retriever.invoke(topic) if topic else retriever.invoke("general overview")[:state['context_chunks']]
    return {"documents": documents, **state}

def get_context_with_sources(documents: List[Document]) -> str:
    return "\n\n".join(f"Source Page: {doc.metadata.get('page', 'N/A')}\nContent: {doc.page_content}" for doc in documents)

def mcq_agent(state: GraphState) -> GraphState:
    context_with_sources = get_context_with_sources(state["documents"])
    prompt = ChatPromptTemplate.from_template(
        """
        **System Instruction:** Your response MUST be a single, raw JSON object. Do not include any conversational text.
        **Your Task:** Generate {num_questions} multiple-choice questions about '{topic}' based on the context.
        **Rules:**
        1. Questions must be factually correct and answerable from the text.
        2. Incorrect options must be plausible but wrong.
        3. Each question MUST include a `source_page` field from the context.
        4. If you cannot create enough valid questions, generate as many as you can.
        **Example:** {{"questions": [{{"question": "What is the primary rule of solving an equation?", "options": ["Move variables left", "Do the same to both sides", "Simplify right side first", "Add before subtracting"], "correct_answer": "Do the same to both sides", "explanation": "The rule is to always do to one side of the equal sign what you do to the other.", "source_page": 4}}]}}
        **Context with Sources:** --- {context} ---
        """
    )
    chain = prompt | llm.with_structured_output(MCQs)
    result = chain.invoke({"context": context_with_sources, "num_questions": state["num_questions"], "topic": state.get("topic", "general topics")})
    return {"final_output": result.dict()}

def fitb_agent(state: GraphState) -> GraphState:
    context_with_sources = get_context_with_sources(state["documents"])
    # THIS IS THE CORRECTED, HIGH-QUALITY PROMPT WITH THE FEW-SHOT EXAMPLE
    prompt = ChatPromptTemplate.from_template(
        """
        **System Instruction:** Your response MUST be a single, raw JSON object. Do not add conversational text.
        **Your Task:** Generate {num_questions} fill-in-the-blank questions about '{topic}' based on the context.
        **Rules:**
        1. Find an important sentence and replace a single key term with '_________'.
        2. The `correct_answer` must be the exact term you removed.
        3. Include the `source_page` for each question.
        4. If you cannot create enough valid questions, return as many as you can.

        **High-Quality Example (from an algebra text):**
        ```json
        {{
          "questions": [
            {{
              "sentence": "An _________ has an 'equal' sign, but an expression does not.",
              "correct_answer": "equation",
              "source_page": 1
            }}
          ]
        }}
        ```
        **Context with Sources:** --- {context} ---
        """
    )
    chain = prompt | llm.with_structured_output(FillInTheBlanks)
    result = chain.invoke({"context": context_with_sources, "num_questions": state["num_questions"], "topic": state.get("topic", "general topics")})
    return {"final_output": result.dict()}

def summary_agent(state: GraphState) -> GraphState:
    context_with_sources = get_context_with_sources(state["documents"])
    source_pages = sorted(list(set(int(doc.metadata.get("page", 0)) for doc in state["documents"])))
    prompt = ChatPromptTemplate.from_template(
        """
        **System Instruction:** Your response MUST be a single, raw JSON object. Do not add conversational text.
        **Your Task:** Generate a high-quality summary of the context, focusing on '{topic}'.
        **Rules:**
        1. Synthesize the key points into a coherent paragraph.
        2. Summary length should be appropriate for the context.
        3. Include the `source_pages` field with all unique page numbers used.
        **Context with Sources:** --- {context} ---
        """
    )
    chain = prompt | llm.with_structured_output(Summary)
    result = chain.invoke({"context": context_with_sources, "topic": state.get("topic", "the main concepts")})
    result.source_pages = source_pages
    return {"final_output": result.dict()}

# --- Router Logic ---
def route_to_agent(state: GraphState) -> str:
    content_type = state['content_type']
    route_map = {
        "MCQ": "mcq_agent",
        "FillInTheBlank": "fitb_agent",
        "Summary": "summary_agent"
    }
    return route_map[content_type]

# --- Build the Graph ---
workflow = StateGraph(GraphState)
workflow.add_node("retriever", retrieve_documents)
workflow.add_node("mcq_agent", mcq_agent)
workflow.add_node("fitb_agent", fitb_agent)
workflow.add_node("summary_agent", summary_agent)
workflow.set_entry_point("retriever")
workflow.add_conditional_edges("retriever", route_to_agent, {
    "mcq_agent": "mcq_agent",
    "fitb_agent": "fitb_agent",
    "summary_agent": "summary_agent",
})
workflow.add_edge("mcq_agent", END)
workflow.add_edge("fitb_agent", END)
workflow.add_edge("summary_agent", END)
app_graph = workflow.compile()

# --- Main Service Function ---
def run_generation(topic: Optional[str], content_type: str, num_questions: Optional[int], context_chunks: int):
    try:
        initial_state = {"topic": topic, "content_type": content_type, "num_questions": num_questions, "context_chunks": context_chunks}
        final_state = app_graph.invoke(initial_state)
        return final_state.get("final_output")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred during content generation: {str(e)}")