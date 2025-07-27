import os
import platform
import multiprocessing

if platform.system() == "Darwin":
    multiprocessing.set_start_method("fork")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from fastapi import FastAPI
from .api.router import api_router

app = FastAPI(
    title="CogniText: Agentic Content Generation Framework",
    description="An AI system to generate MCQs, Fill-in-the-Blanks, and Summaries from PDF files.",
    version="2.0.0"
)

app.include_router(api_router)

@app.get("/", include_in_schema=False)
def root():
    return {"message": "API is running. See /docs for documentation."}