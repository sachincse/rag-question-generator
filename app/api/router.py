from fastapi import APIRouter
from .endpoints import ingest, generate

api_router = APIRouter()
api_router.include_router(ingest.router)
api_router.include_router(generate.router)