import os
import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

TEST_PDF_PATH = "A Quick Algebra Review (1).pdf"

def test_ingest_pdf():
    with open(TEST_PDF_PATH, "rb") as f:
        response = client.post("/ingest", files={"file": (TEST_PDF_PATH, f, "application/pdf")})
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "table_of_contents" in data
    assert isinstance(data["table_of_contents"], list)
    assert any("Exponents" in item for item in data["table_of_contents"])

def test_generate_mcq():
    # Ensure ingest is done first
    with open(TEST_PDF_PATH, "rb") as f:
        client.post("/ingest", files={"file": (TEST_PDF_PATH, f, "application/pdf")})
    payload = {"topic": "Exponents", "content_type": "MCQ"}
    response = client.post("/generate/content", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "questions" in data
    assert isinstance(data["questions"], list)
    assert len(data["questions"]) > 0
    q = data["questions"][0]
    assert "question" in q
    assert "options" in q
    assert "correct_answer" in q
    assert "explanation" in q 