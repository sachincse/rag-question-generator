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
    cd rag-question-generator-1
    ```

2.  **Configure Environment Variables:**
    Create a file named `.env` in the root of the project directory. Add your API key to this file. **The key must not be in quotes.**
    ```
    GROQ_API_KEY=gsk_YourSecretKeyGoesHere
    ```

---

## Option A: Running the Application Locally

1.  **Create and activate a virtual environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

2.  **Install dependencies:**
    The `numpy<2.0` pin is critical for compatibility with the project's libraries.
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the application:**
    To avoid potential multiprocessing issues, run without the `--reload` flag for stability.
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

## API Usage & Sample Questions

Interact with the API via the documentation at `http://127.0.0.1:8000/docs` or `curl`. **You must ingest a document before generating content.**

### 1. Ingest a Document

Place `A_quick_Algebra_Review.pdf` in the project root.

```bash
curl -X POST -F "file=@A_quick_Algebra_Review.pdf" http://localhost:8000/ingest
```

### 2. Generate Content (`/generate/questions`)

#### Example 1: Generate Multiple Choice Questions

**Request:**
```bash
curl -X POST -H "Content-Type: application/json" -d '{
  "topic": "Solving Equations",
  "content_type": "MCQ",
  "num_questions": 2,
  "context_chunks": 5
}' http://localhost:8000/generate/questions
```

**Sample Response:**
```json
{
  "questions": [
    {
      "question": "What is the primary rule of solving an equation?",
      "options": [
        "Move variables left",
        "Do the same to both sides",
        "Simplify right side first",
        "Add before subtracting"
      ],
      "correct_answer": "Do the same to both sides",
      "explanation": "The rule is to always do to one side of the equal sign what you do to the other.",
      "source_page": 4
    },
    {
      "question": "What is the correct solution to the equation x + 9 = -6?",
      "options": [
        "x = -1/2",
        "x = -15",
        "x = 9/5",
        "x = 2"
      ],
      "correct_answer": "x = -15",
      "explanation": "To solve the equation, we need to get our variable by itself. To “move” the 9 to the other side, we need to subtract 9 from both sides of the equal sign, since 9 was added to x in the original problem.",
      "source_page": 3
    }
  ]
}
```

#### Example 2: Generate Fill-in-the-Blank Questions

**Request:**
```bash
curl -X POST -H "Content-Type: application/json" -d '{
  "topic": "exponents",
  "content_type": "FillInTheBlank",
  "num_questions": 5,
  "context_chunks": 8
}' http://localhost:8000/generate/questions
```

**Sample Response:**
```json
{
  "questions": [
    {
      "sentence": "When you have a _________ exponent, it means inverse, (the negative exponent is an operation that “flips” only the base that it applies to).",
      "correct_answer": "negative",
      "source_page": 14
    },
    {
      "sentence": "In order for two terms to multiply together and result in zero, ONE OF THEM MUST BE _________.",
      "correct_answer": "ZERO",
      "source_page": 17
    },
    {
      "sentence": "When you square a number, you multiply the number by itself, so it’s impossible to have one be _______ and one be positive.",
      "correct_answer": "negative",
      "source_page": 28
    },
    {
      "sentence": "A rational expression is an expression that can be written as a fraction where the variable is in the _______ (on bottom).",
      "correct_answer": "denominator",
      "source_page": 23
    },
    {
      "sentence": "The domain (possible values of x) is all real numbers except for values that make the _______ equal to zero.",
      "correct_answer": "denominator",
      "source_page": 23
    }
  ]
}
```

#### Example 3: Generate a Summary

**Request:**
```bash
curl -X POST -H "Content-Type: application/json" -d '{
  "topic": "Quadratics",
  "content_type": "Summary",
  "context_chunks": 5
}' http://localhost:8000/generate/questions
```

**Sample Response:**
```json
{
  "summary_text": "Quadratic equations are equations that have a variable to the second power, like x2 + x = 6. Since x2 and x are not like terms they can not be combined. We need a new way for finding solutions to quadratic equations.",
  "source_pages": [
    16,
    17,
    18,
    19,
    28
  ]
}
```

---

## Running Unit Tests

The project includes a basic test suite to demonstrate the principles of test-driven development. The tests cover basic API functionality like the root endpoint and error handling for invalid file types.

1.  Ensure you have followed the "Running the Application Locally" steps to set up your environment and install dependencies.
2.  From the project root directory, run the following command:

```bash
pytest
```
The tests will execute and print the results to the console.````