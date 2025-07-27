from fastapi import FastAPI, UploadFile, File, HTTPException
from .models.schemas import IngestResponse, ContentGenerationRequest
from .services import document_service, qg_service

# Initialize the FastAPI app with metadata
app = FastAPI(
    title="Agent Framework for Content Generation",
    description="An AI system to generate MCQs, Fill-in-the-Blanks, and Summaries from PDF files using a multi-agent workflow.",
    version="1.0.0"
)

@app.post("/ingest", response_model=IngestResponse)
async def ingest_pdf(file: UploadFile = File(..., description="The PDF file to be processed.")):
    """
    Uploads and processes a PDF file. This endpoint:
    - Validates that the file is a PDF.
    - Saves the file to disk.
    - Chunks the text and ingests it into a FAISS vector database.
    - Returns the document's table of contents.
    
    This must be called successfully before using the /generate/content endpoint.
    """
    if file.content_type != 'application/pdf':
        raise HTTPException(status_code=400, detail="Invalid file type. Only PDF files are allowed.")
    
    # Delegate the core logic to the document service
    result = await document_service.process_and_ingest_pdf(file)
    return result

@app.post("/generate/content", response_model=dict)
def generate_content(request: ContentGenerationRequest):
    """
    Generates content based on a topic from the ingested PDF. This endpoint:
    - Takes a topic and a content_type ('MCQ', 'FillInTheBlank', or 'Summary').
    - Orchestrates a multi-agent workflow that retrieves relevant documents
      and routes to a specialized agent for generation.
    - Returns the generated content as a JSON object.
    - For MCQ and FillInTheBlank, the response also includes an evaluation of the generated questions.
    """
    try:
        # Delegate the core logic to the question generation service
        generated_data = qg_service.run_generation(request.topic, request.content_type)
        if not generated_data:
            raise HTTPException(status_code=404, detail="The agent could not generate content for the given topic. Please try another topic.")
        return generated_data
    except FileNotFoundError as e:
        # This custom exception is raised if the vector store doesn't exist
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        # Catch any other unexpected errors during the process
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

@app.get("/", include_in_schema=False)
def root():
    """A simple root endpoint to confirm the API is running."""
    return {"message": "Agentic Content Generation API is running. See /docs for API documentation."}