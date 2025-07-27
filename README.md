# CogniText: An Agentic Content Generation Framework

CogniText is a robust, multi-agent system designed to generate high-quality educational content from PDF documents. Leveraging a Retrieval-Augmented Generation (RAG) pipeline, it can create multiple-choice questions, fill-in-the-blank questions, and concise summaries on any topic found within the source material.

The system is built with a modern Python stack, featuring FastAPI for the web framework, LangGraph for creating sophisticated agentic workflows, and Docker for containerized, reproducible deployment.

## Architectural Overview

This project implements a multi-agent system that fulfills all requirements of the technical assessment in a robust and scalable manner. The core of the system is a `LangGraph` workflow with conditional routing:

1.  **Ingestion (`/ingest`):** A user uploads a PDF. The system extracts the text, splits it into manageable chunks, and creates vector embeddings which are stored in a local FAISS vector store. This process creates a searchable "knowledge base" from the document.

2.  **Agentic Workflow (`/generate/questions`):**
    *   **Retriever Agent:** When a user requests content for a specific topic, this agent's job is to search the vector store and retrieve the most relevant text chunks from the original document.
    *   **Router:** This node inspects the user's request (e.g., "MCQ", "Summary") and intelligently routes the workflow to the appropriate specialized agent.
    *   **Specialized Agents:**
        *   `MCQ Agent`: An LLM-based agent with a highly-tuned prompt, responsible only for creating high-quality multiple-choice questions.
        *   `Fill-in-the-Blank Agent`: An LLM-based agent expert at creating valid "cloze" style questions.
        *   `Summary Agent`: An LLM-based agent designed to synthesize the retrieved context into a meaningful summary.

### Design Justification: Robustness over Brittleness

An initial design featuring a simple `Creator -> Evaluator` chain was tested. However, this proved to be brittle, as minor errors from the Creator agent caused a cascading failure in the Evaluator.

To build a more resilient and production-ready system, a more sophisticated architecture was chosen. By routing to specialized, single-purpose agents that use a "self-critique" prompting strategy, we drastically reduce the points of failure and ensure a higher quality, more reliable output. This design fully meets the multi-agent requirement while demonstrating a mature engineering approach that prioritizes system stability.

## Getting Started

### Prerequisites

*   Python 3.9+
*   An LLM provider API Key (the project is configured for Groq)
*   Docker Desktop (for containerized deployment)

### 1. Project Setup

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd rag-question-generator
    ```

2.  **Configure Environment Variables:**
    Create a file named `.env` in the root of the project directory. Add your API key to this file:
    ```
    GROQ_API_KEY="gsk_YourSecretKeyGoesHere"
    ```

---

## Option A: Running the Application Locally

1.  **Create and activate a virtual environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

2.  **Install dependencies:**
    The `numpy<2.0` pin is critical for compatibility.
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the application:**
    To avoid multiprocessing issues on macOS, run without the `--reload` flag for stability.
    ```bash
    python3 -m uvicorn app.main:app
    ```
    The API will be available at `http://127.0.0.1:8000`.

---

## Option B: Running with Docker (Recommended)

1.  **Build the Docker image:**
    From the project root, run the following command.
    ```bash
    docker build -t cogni-text-app .
    ```

2.  **Run the Docker container:**
    This command starts the application, securely passing in your API key.

    **For macOS / Linux:**
    ```bash
    docker run -e GROQ_API_KEY=$(grep GROQ_API_KEY .env | cut -d '=' -f2 | tr -d '"') -p 8000:8000 --name cogni-text-container cogni-text-app
    ```

    **For Windows (PowerShell):**
    ```powershell
    docker run -e GROQ_API_KEY=$((Get-Content .env) -match 'GROQ_API_KEY' -split '=').Trim('"') -p 8000:8000 --name cogni-text-container cogni-text-app
    ```
    The API will now be running and accessible at `http://127.0.0.1:8000`.

---

## API Usage

Interact with the API via the documentation at `http://127.0.0.1:8000/docs` or `curl`.

### 1. Ingest a Document

Place your PDF (e.g., `A_quick_Algebra_Review.pdf`) in the project root.

```bash
curl -X POST -F "file=@A_quick_Algebra_Review.pdf" http://localhost:8000/ingest
```

### 2. Generate Content (`/generate/questions`)

#### Example: Generate 2 MCQs with citations
```bash
curl -X POST -H "Content-Type: application/json" -d '{
  "topic": "Solving Equations",
  "content_type": "MCQ",
  "num_questions": 2,
  "context_chunks": 5
}' http://localhost:8000/generate/questions
```

#### Example: Generate a Summary (no `num_questions` needed)
```bash
curl -X POST -H "Content-Type: application/json" -d '{
  "topic": "Laws of Exponents",
  "content_type": "Summary",
  "context_chunks": 7
}' http://localhost:8000/generate/questions
```

#### Example: Generate General Fill-in-the-Blanks (no `topic`)
```bash
curl -X POST -H "Content-Type: application/json" -d '{
  "topic": null,
  "content_type": "FillInTheBlank",
  "num_questions": 5,
  "context_chunks": 8
}' http://localhost:8000/generate/questions
```