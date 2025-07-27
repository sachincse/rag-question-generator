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

# --- Agent Nodes (with Final, Production-Grade Prompts) ---
def retrieve_documents(state: GraphState) -> GraphState:
    # This function remains unchanged
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
        documents = retriever.invoke("general overview of the document's main concepts")[:state['context_chunks']]

    return {"documents": documents, **state}

def get_context_with_sources(documents: List[Document]) -> str:
    # This function remains unchanged
    return "\n\n".join(
        f"Source Page: {doc.metadata.get('page', 'N/A')}\nContent: {doc.page_content}"
        for doc in documents
    )

def mcq_agent(state: GraphState) -> GraphState:
    print("--- Node: MCQ Agent ---")
    context_with_sources = get_context_with_sources(state["documents"])
    
    prompt = ChatPromptTemplate.from_template(
        """
        **System Instruction:**
        - Your response MUST be a single, raw JSON object. Do not add conversational text, prefixes, or markdown.

        **Your Task:**
        - You are an expert question designer and subject matter expert.
        - Generate {num_questions} high-quality multiple-choice questions based on the `Context with Sources` provided.

        **Rules & Constraints:**
        1.  **Topic Focus:** Generate questions STRICTLY about the user's topic: '{topic}'. Ignore any unrelated information found in the context.
        2.  **Factual Accuracy:** Each question, its options, and its explanation must be factually correct and directly derivable from the provided text.
        3.  **Plausible Distractors:** The incorrect options should be plausible but clearly wrong based on the context.
        4.  **Citations:** Every question MUST include a `source_page` field, referencing the page number provided in the context.
        5.  **Graceful Failure:** If you cannot create {num_questions} high-quality questions that follow all rules, generate as many as you can and stop. Do not fabricate content.

        **High-Quality Example (from an algebra text):**
        ```json
        {{
          "questions": [
            {{
              "question": "What is the primary rule when solving an equation?",
              "options": [
                "Always move variables to the left side",
                "Perform multiplication before addition",
                "Whatever you do to one side, you must do to the other",
                "Simplify the right side first"
              ],
              "correct_answer": "Whatever you do to one side, you must do to the other",
              "explanation": "The document states the important rule is to always do to one side of the equal sign what you do to the other.",
              "source_page": 4
            }}
          ]
        }}
        ```

        **Context with Sources:**
        ---
        {context}
        ---
        """
    )
    chain = prompt | llm.with_structured_output(MCQs)
    result = chain.invoke({"context": context_with_sources, "num_questions": state["num_questions"], "topic": state.get("topic", "general topics")})
    return {"final_output": result.dict()}

def fitb_agent(state: GraphState) -> GraphState:
    print("--- Node: Fill-in-the-Blank Agent ---")
    context_with_sources = get_context_with_sources(state["documents"])

    prompt = ChatPromptTemplate.from_template(
        """
        **System Instruction:**
        - Your response MUST be a single, raw JSON object. Do not add conversational text.

        **Your Task:**
        - You are an expert at creating educational assessments.
        - Generate {num_questions} high-quality fill-in-the-blank questions based on the `Context with Sources`.

        **Rules & Constraints:**
        1.  **Topic Focus:** Generate questions STRICTLY about the user's topic: '{topic}'.
        2.  **Sentence Selection:** For each question, select a complete, informative sentence from the context.
        3.  **Keyword Replacement:** Identify a single, critical keyword or short phrase in that sentence. Replace ONLY that keyword/phrase with '_________'.
        4.  **Answer Accuracy:** The `correct_answer` field must contain the exact keyword/phrase you removed.
        5.  **Citations:** Each question MUST include the `source_page`.
        6.  **Graceful Failure:** If you cannot create enough valid questions, return only as many as you can.

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
    print("--- Node: Summary Agent ---")
    context_with_sources = get_context_with_sources(state["documents"])
    source_pages = sorted(list(set(doc.metadata.get("page", 0) for doc in state["documents"])))
    
    prompt = ChatPromptTemplate.from_template(
        """
        **System Instruction:**
        - Your response MUST be a single, raw JSON object. Do not add conversational text.

        **Your Task:**
        - You are an expert academic writer and information synthesizer.
        - Generate a meaningful, high-quality summary of the provided `Context with Sources`.

        **Rules & Constraints:**
        1.  **Synthesize, Don't List:** Your summary should synthesize the key points into a coherent paragraph. Do NOT just list the topics found. Explain the main ideas and how they relate to each other.
        2.  **Topic Focus:** The summary should focus on the user's topic: '{topic}'.
        3.  **Dynamic Length:** The length of the summary should be proportional to the amount of context provided. A short context requires a short summary; a longer, more detailed context may require a longer summary to be meaningful.
        4.  **Citations:** The final JSON must include the `source_pages` field, listing all unique page numbers used to create the summary.

        **High-Quality Example (from an algebra text):**
        ```json
        {{
          "summary_text": "The process of simplifying expressions is fundamentally about combining like terms to make the expression as concise as possible, which is governed by the order of operations (PEMDAS). This is distinct from solving equations, which are identifiable by their equal sign and have the goal of isolating a variable by performing inverse operations on both sides.",
          "source_pages":
        }}
        ```
        
        **Context with Sources:**
        ---
        {context}
        ---
        """
    )
    chain = prompt | llm.with_structured_output(Summary)
    result = chain.invoke({"context": context_with_sources, "topic": state.get("topic", "the main concepts")})
    result.source_pages = source_pages
    return {"final_output": result.dict()}


# --- Router, Validation, and Graph Build (All Unchanged) ---
def route_to_agent(state: GraphState) -> str:
    route_map = {"MCQ": "mcq_agent", "FillInTheBlank": "fitb_agent", "Summary": "summary_agent"}
    return route_map[state['content_type']]

def validate_and_filter_output(output: dict, topic: Optional[str]) -> dict:
    if "questions" not in output: return output
    validated_questions = []
    topic_word = topic.lower().split(" ")[-1] if topic else None
    for q in output["questions"]:
        if "sentence" in q and "_________" not in q.get("sentence", ""): continue
        if not q.get("correct_answer"): continue
        if topic_word:
            combined_text = (q.get("question", "") + q.get("explanation", "") + q.get("sentence", "")).lower()
            if topic_word not in combined_text: continue
        validated_questions.append(q)
    output["questions"] = validated_questions
    return output

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

def run_generation(topic: Optional[str], content_type: Literal["MCQ", "FillInTheBlank", "Summary"], num_questions: Optional[int], context_chunks: int):
    try:
        initial_state = {
            "topic": topic, "content_type": content_type,
            "num_questions": num_questions, "context_chunks": context_chunks
        }
        final_state = app_graph.invoke(initial_state)
        generated_output = final_state.get("final_output")
        validated_output = validate_and_filter_output(generated_output, topic)
        return validated_output
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred during content generation: {str(e)}")