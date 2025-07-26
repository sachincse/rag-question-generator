from typing import List, Literal
from pydantic import BaseModel, Field

# --- API Request/Response Models ---
class IngestResponse(BaseModel):
    message: str
    table_of_contents: List[str]

class ContentGenerationRequest(BaseModel):
    topic: str
    content_type: Literal["MCQ", "FillInTheBlank", "Summary"]

# --- Agent Output Models (must match the logic in qg_service.py) ---
class MCQ(BaseModel):
    question: str
    options: List[str]
    correct_answer: str
    explanation: str

class MCQs(BaseModel):
    questions: List[MCQ]

class FillInTheBlank(BaseModel):
    sentence: str
    correct_answer: str

class FillInTheBlanks(BaseModel):
    questions: List[FillInTheBlank]

class Summary(BaseModel):
    summary_text: str