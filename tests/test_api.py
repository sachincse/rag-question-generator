import pytest
from fastapi.testclient import TestClient
from app.main import app
import os
import shutil # <--- Import the shutil library

@pytest.fixture(scope="module")
def client():
    """Create a test client for the API that is reused for all tests in this module."""
    return TestClient(app)

def test_root(client):
    """Test the root endpoint to ensure the API is running."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "API is running. See /docs for documentation."}

def test_ingest_non_pdf(client):
    """Test that uploading a non-PDF file results in a 400 Bad Request error."""
    # Create a dummy text file for the test
    with open("test_file.txt", "w") as f:
        f.write("this is not a pdf")
    
    with open("test_file.txt", "rb") as f:
        response = client.post("/ingest", files={"file": ("test_file.txt", f, "text/plain")})
    
    assert response.status_code == 400
    assert "Invalid file type" in response.json()["detail"]
    
    # Clean up the dummy file
    os.remove("test_file.txt")

def test_generate_before_ingest(client):
    """Test that the generate endpoint returns an error if no document has been ingested."""
    # Define the path to the vector store directory
    vector_store_dir = "vector_store"

    # --- THIS IS THE CORRECTED CLEANUP LOGIC ---
    # Ensure the entire vector_store directory is removed before the test
    if os.path.exists(vector_store_dir):
        # Use shutil.rmtree() to recursively delete the directory and its contents
        shutil.rmtree(vector_store_dir)
        
    request_data = {
        "topic": "testing",
        "content_type": "MCQ",
        "num_questions": 3,
        "context_chunks": 5
    }
    response = client.post("/generate/questions", json=request_data)
    
    assert response.status_code == 400
    assert "Vector store not found" in response.json()["detail"]