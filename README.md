# Agent Framework for RAG-Based Question Generation and Summarization

## Overview
This project is a multi-agent system for generating MCQ and fill-in-the-blank questions and summaries from PDF files using Retrieval Augmented Generation (RAG) and Large Language Models (LLMs).

## Features
- Ingest PDF files and extract Table of Contents
- Generate MCQ, Fill-in-the-Blank questions, and summaries using LLMs
- Multi-agent orchestration (retriever, generator, evaluator)
- FastAPI endpoints
- Dockerized for easy deployment

## Requirements
- Python 3.10+
- Docker (optional, for containerized deployment)
- GROQ_API_KEY (for LLM access, set in `.env` file)

## Setup

### 1. Clone the repository
```
git clone <repo-url>
cd rag-question-generator
```

### 2. Install dependencies
```
pip install -r requirements.txt
```

### 3. Set up environment variables
Create a `.env` file in the root directory:
```
GROQ_API_KEY=your_groq_api_key_here
```

### 4. Run the app
```
uvicorn app.main:app --reload
```
Visit [http://localhost:8000/docs](http://localhost:8000/docs) for the interactive API docs.

### 5. Run with Docker
```
docker build -t rag-qg .
docker run -p 8000:8000 --env-file .env rag-qg
```

## API Endpoints

### 1. `/ingest` (POST)
- Upload a PDF file.
- Returns: Table of Contents.

### 2. `/generate/content` (POST)
- Input: `{ "topic": "<topic>", "content_type": "MCQ" | "FillInTheBlank" | "Summary" }`
- Returns: Generated questions or summary.

## Functional Test Example

### Upload and Ingest
```
POST /ingest
file: A Quick Algebra Review (1).pdf
Response:
{
  "message": "File 'A Quick Algebra Review (1).pdf' processed successfully.",
  "table_of_contents": [
    "1. Introduction",
    "2. Real Numbers",
    "3. Exponents and Radicals",
    ...
  ]
}
```

### Generate MCQ Questions
```
POST /generate/content
{
  "topic": "Exponents",
  "content_type": "MCQ"
}
Response:
{
  "questions": [
    {
      "question": "What does a negative exponent indicate?",
      "options": ["Multiplication", "Division", "Reciprocal", "Addition"],
      "correct_answer": "Reciprocal",
      "explanation": "A negative exponent means take the reciprocal of the base."
    },
    ...
  ]
}
```

## Testing

### Run Functional Tests
```
pytest
```

---

For any questions, email technicaltest@alefeducation.com. 