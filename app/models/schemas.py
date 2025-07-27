from typing import List, Literal, Optional
from pydantic import BaseModel, Field

# API Request/Response Models
class IngestResponse(BaseModel):
    message: str
    table_of_contents: List[str]

class QuestionGenerationRequest(BaseModel):
    topic: Optional[str] = Field(None, description="Topic for content generation. Omit for general questions.")
    content_type: Literal["MCQ", "FillInTheBlank", "Summary"]
    num_questions: Optional[int] = Field(3, description="Desired number of questions (for MCQ/FillInTheBlank).", gt=0, le=10)
    context_chunks: int = Field(5, description="Number of document chunks to use as context.", gt=0, le=15)

# Agent Output Models
class MCQ(BaseModel):
    question: str
    options: List[str]
    correct_answer: str
    explanation: str
    source_page: int = Field(description="Page number in the PDF for reference.")

class MCQs(BaseModel):
    questions: List[MCQ]

class FillInTheBlank(BaseModel):
    sentence: str
    correct_answer: str
    source_page: int = Field(description="Page number in the PDF for reference.")

class FillInTheBlanks(BaseModel):
    questions: List[FillInTheBlank]

class Summary(BaseModel):
    summary_text: str
    source_pages: List[int] = Field(description="List of page numbers used for the summary.")