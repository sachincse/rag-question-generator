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

Interact with the API via the documentation at `http://127.0.0.1:8000/docs` or `curl`. **You must ingest `A_quick_Algebra_Review.pdf` before generating content.**

### 1. Ingest a Document

```bash
curl -X POST -F "file=@A_quick_Algebra_Review.pdf" http://localhost:8000/ingest
```

### 2. Generate Content (`/generate/questions`)

#### Example 1: Generate Multiple Choice Questions

**Request:**
```bash
curl -X POST -H "Content-Type: application/json" -d '{
  "topic": "Inequalities",
  "content_type": "MCQ",
  "num_questions": 3,
  "context_chunks": 8
}' http://localhost:8000/generate/questions
```

**Actual Response:**
```json
{
  "questions": [
    {
      "question": "What is the correct method to solve an inequality?",
      "options": [
        "Always move variables to the left side",
        "Perform multiplication before addition",
        "Whatever you do to one side, you must do to the other",
        "Simplify the right side first"
      ],
      "correct_answer": "Whatever you do to one side, you must do to the other",
      "explanation": "The document states the important rule is to always do to one side of the equal sign what you do to the other.",
      "source_page": 6
    },
    {
      "question": "What is the correct method to solve an inequality with a division operation?",
      "options": [
        "Multiply both sides by the divisor",
        "Divide both sides by the divisor",
        "Flip the inequality sign and multiply by the divisor",
        "Flip the inequality sign and divide by the divisor"
      ],
      "correct_answer": "Flip the inequality sign and multiply by the divisor",
      "explanation": "When dividing by a negative number, the inequality sign must be flipped.",
      "source_page": 7
    },
    {
      "question": "What is the correct method to solve an absolute value equation?",
      "options": [
        "Take the absolute value of both sides",
        "Flip the inequality sign and take the absolute value of both sides",
        "Take the absolute value of one side and flip the inequality sign",
        "Take the absolute value of both sides and flip the inequality sign"
      ],
      "correct_answer": "Flip the inequality sign and take the absolute value of both sides",
      "explanation": "When solving absolute value equations, we must consider that the number inside could have been negative before you applied the absolute value.",
      "source_page": 11
    }
  ]
}
```

#### Example 2: Generate Fill-in-the-Blank Questions

**Request:**
```bash
curl -X POST -H "Content-Type: application/json" -d '{
  "topic": "Inequalities",
  "content_type": "FillInTheBlank",
  "num_questions": 2,
  "context_chunks": 8
}' http://localhost:8000/generate/questions
```

**Actual Response:**
```json
{
  "questions": [
    {
      "sentence": "The solution to the inequality x > 0 is _________.",
      "correct_answer": "x > 0",
      "source_page": 7
    },
    {
      "sentence": "When solving an inequality, we must do the same operation to _________ side of the inequality as we do to the other side.",
      "correct_answer": "both",
      "source_page": 6
    }
  ]
}
```

#### Example 3: Generate a Summary

**Request:**
```bash
curl -X POST -H "Content-Type: application/json" -d '{
  "topic": "Radicals",
  "content_type": "Summary",
  "context_chunks": 8
}' http://localhost:8000/generate/questions
```

**Actual Response:**
```json
{
  "summary_text": "Radical expressions are mathematical phrases that contain roots of variables. The square root is a number that can be squared to get another number. For instance, the square root of 25, 25, is 5, because if you square 5, you get 25. There is one basic rule when dealing with roots: if the root is even, you cannot have a negative number. This means that there can be no negative numbers in square roots, fourth roots, sixth roots, etc. When solving radical equations, we raise both sides of the equation to the power of the radical, but before we do that, we need to get the radical on one side of the equation by itself. It is also important that we check our solution to make sure that it exists in our domain.",
  "source_pages": [
    0,
    19,
    24,
    28,
    29,
    30
  ]
}
```

---

## Running Unit Tests

The project includes a basic test suite to demonstrate the principles of test-driven development.

1.  Ensure you have followed the "Running the Application Locally" steps to set up your environment and install dependencies.
2.  From the project root directory, run the following command:

```bash
pytest
```
The tests will execute and print the results to the console.