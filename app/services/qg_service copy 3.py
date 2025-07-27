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
    # ... (This function remains unchanged)
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
        documents = retriever.invoke(topic)
    else:
        documents = retriever.invoke("general python programming concepts")[:state['context_chunks']]

    return {"documents": documents, **state}

def get_context_with_sources(documents: List[Document]) -> str:
    # ... (This function remains unchanged)
    return "\n\n".join(
        f"Source Page: {doc.metadata.get('page', 'N/A')}\nContent: {doc.page_content}"
        for doc in documents
    )

def mcq_agent(state: GraphState) -> GraphState:
    # ... (MCQ agent remains the same)
    print("--- Node: MCQ Agent ---")
    context_with_sources = get_context_with_sources(state["documents"])
    prompt = ChatPromptTemplate.from_template(
        """**Task:** Based on the context below, generate {num_questions} high-quality multiple-choice questions strictly about the topic '{topic}'. Ignore unrelated information in the context. Your response must be a single, raw JSON object. Each question must include a `source_page` field. If you cannot generate enough high-quality questions, generate as many as you can.
        **Context with Sources:** --- {context} --- """
    )
    chain = prompt | llm.with_structured_output(MCQs)
    result = chain.invoke({"context": context_with_sources, "num_questions": state["num_questions"], "topic": state.get("topic", "general topics")})
    return {"final_output": result.dict()}

def fitb_agent(state: GraphState) -> GraphState:
    print("--- Node: Fill-in-the-Blank Agent ---")
    context_with_sources = get_context_with_sources(state["documents"])
    # New, highly explicit prompt with step-by-step instructions
    prompt = ChatPromptTemplate.from_template(
        """**Task:** Create {num_questions} fill-in-the-blank questions based on the context. Your response must be a single, raw JSON object.
        Follow these steps precisely for each question:
        1.  Find an important, factual sentence in the context that is clearly about the topic '{topic}'.
        2.  Identify a single, critical keyword or short phrase in that sentence.
        3.  Create the "sentence" field by replacing that keyword with '_________'.
        4.  Create the "correct_answer" field with the exact keyword you removed.
        5.  Add the correct "source_page" from the context.

        If you cannot create {num_questions} high-quality questions that follow these rules, create as many as you can.

        **Context with Sources:**
        ---
        {context}
        ---
        """
    )
    chain = prompt | llm.with_structured_output(FillInTheBlanks)
    result = chain.invoke({"context": context_with_sources, "num_questions": state["num_questions"], "topic": state.get("topic", "general topics")})
    return {"final_output": result.dict()}

def summary_agent(state: GraphState) -> GraphState:
    # ... (Summary agent remains the same)
    print("--- Node: Summary Agent ---")
    context_with_sources = get_context_with_sources(state["documents"])
    source_pages = sorted(list(set(doc.metadata.get("page", 0) for doc in state["documents"])))
    prompt = ChatPromptTemplate.from_template(
        """**Task:** Generate a concise summary of the context. Your response must be a single, raw JSON object.
        **Context with Sources:** --- {context} --- """
    )
    chain = prompt | llm.with_structured_output(Summary)
    result = chain.invoke({"context": context_with_sources})
    result.source_pages = source_pages
    return {"final_output": result.dict()}

# --- Router Logic (Unchanged) ---
def route_to_agent(state: GraphState) -> str:
    route_map = {"MCQ": "mcq_agent", "FillInTheBlank": "fitb_agent", "Summary": "summary_agent"}
    return route_map[state['content_type']]

# --- NEW: Programmatic Validation and Filtering ---
def validate_and_filter_output(output: dict, topic: Optional[str]) -> dict:
    """
    Acts as a final, programmatic quality gate to enforce rules and relevance.
    """
    if "questions" not in output:
        # This is a summary, return as-is
        return output

    validated_questions = []
    topic_word = topic.lower().split(" ")[-1] if topic else None

    for q in output["questions"]:
        # Rule 1: Check for fill-in-the-blank format if applicable
        if "sentence" in q and "_________" not in q.get("sentence", ""):
            continue # Discard if it's not a proper fill-in-the-blank

        # Rule 2: Check for a meaningful answer
        if not q.get("correct_answer"):
            continue # Discard if the answer is empty

        # Rule 3: Check for relevance if a topic was provided
        if topic_word:
            combined_text = (q.get("question", "") + q.get("explanation", "") + q.get("sentence", "")).lower()
            if topic_word not in combined_text:
                continue # Discard if the question is not on topic
        
        validated_questions.append(q)

    output["questions"] = validated_questions
    return output

# --- Build the Graph (Unchanged) ---
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

# --- Main Service Function (Updated signature) ---
def run_generation(topic: Optional[str], content_type: Literal["MCQ", "FillInTheBlank", "Summary"], num_questions: Optional[int], context_chunks: int):
    # ... (the rest of the function is the same)
    try:
        initial_state = {
            "topic": topic,
            "content_type": content_type,
            "num_questions": num_questions,
            "context_chunks": context_chunks,
        }
        final_state = app_graph.invoke(initial_state)
        generated_output = final_state.get("final_output")
        
        validated_output = validate_and_filter_output(generated_output, topic)
        
        return validated_output
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred during content generation: {str(e)}")