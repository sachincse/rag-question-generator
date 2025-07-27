from typing import List, Literal, Optional
from pydantic import BaseModel, Field

# --- API Request/Response Models ---
class IngestResponse(BaseModel):
    message: str
    table_of_contents: List[str]

class ContentGenerationRequest(BaseModel):
    topic: Optional[str] = Field(None, description="The topic for content generation. If omitted, general content from the document will be created.")
    content_type: Literal["MCQ", "FillInTheBlank", "Summary"]
    
    # num_questions is now optional
    num_questions: Optional[int] = Field(3, description="The desired number of questions to generate (used for MCQ and FillInTheBlank).", gt=0, le=10)
    
    context_chunks: int = Field(5, description="Number of document chunks to use as context.", gt=0, le=15)

# --- Agent Output Models (with source citations) ---
class MCQ(BaseModel):
    question: str
    options: List[str]
    correct_answer: str
    explanation: str
    source_page: int = Field(description="The page number in the PDF where the answer can be found.")

class MCQs(BaseModel):
    questions: List[MCQ]

class FillInTheBlank(BaseModel):
    sentence: str
    correct_answer: str
    source_page: int = Field(description="The page number in the PDF where the answer can be found.")

class FillInTheBlanks(BaseModel):
    questions: List[FillInTheBlank]

class Summary(BaseModel):
    summary_text: str
    source_pages: List[int] = Field(description="A list of page numbers used to create the summary.")