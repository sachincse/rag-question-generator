import re
from pathlib import Path
from fastapi import UploadFile, HTTPException
from langchain_community.document_loaders import PyPDFLoader      # Corrected import
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings # Corrected import
from langchain_community.vectorstores import FAISS               # Corrected import

# Define persistent storage directories
UPLOAD_DIR = Path("uploads")
VECTOR_STORE_DIR = Path("vector_store")
UPLOAD_DIR.mkdir(exist_ok=True)
VECTOR_STORE_DIR.mkdir(exist_ok=True)
VECTOR_STORE_PATH = str(VECTOR_STORE_DIR / "algebra_review.faiss")

def get_embeddings_model():
    """Loads the embedding model."""
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

async def process_and_ingest_pdf(file: UploadFile):
    """Handles the logic for ingesting a PDF file."""
    file_path = UPLOAD_DIR / file.filename
    try:
        with open(file_path, "wb") as buffer:
            buffer.write(await file.read())
    except Exception:
        raise HTTPException(status_code=500, detail="Could not save uploaded file.")

    loader = PyPDFLoader(str(file_path))
    docs = loader.load()

    toc = []
    if docs:
        first_page_text = docs[0].page_content
        for line in first_page_text.split('\n'):
            if re.match(r'^\d{1,2}\.\s', line):
                toc.append(line.strip())

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    if not splits:
        raise HTTPException(status_code=400, detail="Could not extract text from the PDF.")

    try:
        embeddings = get_embeddings_model()
        vector_store = FAISS.from_documents(documents=splits, embedding=embeddings)
        vector_store.save_local(VECTOR_STORE_PATH)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create vector store: {e}")

    return {"message": f"File '{file.filename}' processed successfully.", "table_of_contents": toc}