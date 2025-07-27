from fastapi import FastAPI, UploadFile, File, HTTPException
from .models.schemas import IngestResponse, ContentGenerationRequest
from .services import document_service, qg_service

app = FastAPI(
    title="Robust Agent Framework for Content Generation",
    description="An AI system to generate content from PDF files with dynamic controls and source citations.",
    version="2.0.0" # Version up!
)

@app.post("/ingest", response_model=IngestResponse)
async def ingest_pdf(file: UploadFile = File(...)):
    # ... (No changes to this endpoint)
    if file.content_type != 'application/pdf':
        raise HTTPException(status_code=400, detail="Invalid file type. Only PDF files are allowed.")
    result = await document_service.process_and_ingest_pdf(file)
    return result

@app.post("/generate/content", response_model=dict)
def generate_content(request: ContentGenerationRequest):
    """
    Generates content based on a topic from the ingested PDF.

    - **topic**: (Optional) The specific topic to focus on.
    - **content_type**: 'MCQ', 'FillInTheBlank', or 'Summary'.
    - **num_questions**: How many questions to generate (default 3).
    - **context_chunks**: How many document chunks to use for context (default 5).
    """
    try:
        # Pass all parameters from the request to the service
        generated_data = qg_service.run_generation(
            topic=request.topic,
            content_type=request.content_type,
            num_questions=request.num_questions,
            context_chunks=request.context_chunks
        )
        if not generated_data or not any(generated_data.values()):
             raise HTTPException(status_code=404, detail="The agent could not generate content for the given topic/parameters.")
        return generated_data
    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

@app.get("/", include_in_schema=False)
def root():
    return {"message": "Robust Agentic Content Generation API is running. See /docs for API documentation."}