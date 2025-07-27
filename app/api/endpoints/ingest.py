from fastapi import APIRouter, UploadFile, File, HTTPException
from ...models.schemas import IngestResponse
from ...services import document_service

router = APIRouter()

@router.post("/ingest", response_model=IngestResponse, tags=["Ingestion"])
async def ingest_pdf(file: UploadFile = File(...)):
    if file.content_type != 'application/pdf':
        raise HTTPException(status_code=400, detail="Invalid file type. Only PDF files are allowed.")
    return await document_service.process_and_ingest_pdf(file)