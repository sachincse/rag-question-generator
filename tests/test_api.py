import pytest
from fastapi.testclient import TestClient
from app.main import app
import os

@pytest.fixture(scope="module")
def client():
    return TestClient(app)

def test_root(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "API is running. See /docs for documentation."}

def test_ingest_non_pdf(client):
    with open("test_file.txt", "w") as f:
        f.write("hello")
    with open("test_file.txt", "rb") as f:
        response = client.post("/ingest", files={"file": ("test_file.txt", f, "text/plain")})
    assert response.status_code == 400
    assert "Invalid file type" in response.json()["detail"]
    os.remove("test_file.txt")

def test_generate_before_ingest(client):
    # Ensure no vector store exists before this test
    if os.path.exists("vector_store/docs.faiss"):
        # This is a simple cleanup for the test. In a real suite, you'd use a test-specific path.
        os.remove("vector_store/docs.faiss")
        os.remove("vector_store/docs.pkl")
        
    request_data = {
        "topic": "testing",
        "content_type": "MCQ"
    }
    response = client.post("/generate/questions", json=request_data)
    assert response.status_code == 400
    assert "Vector store not found" in response.json()["detail"]