from fastapi import APIRouter, HTTPException
from ...models.schemas import QuestionGenerationRequest
from ...services import qg_service

router = APIRouter()

@router.post("/generate/questions", response_model=dict, tags=["Generation"])
def generate_questions(request: QuestionGenerationRequest):
    try:
        generated_data = qg_service.run_generation(
            topic=request.topic,
            content_type=request.content_type,
            num_questions=request.num_questions,
            context_chunks=request.context_chunks
        )
        if not generated_data or not any(generated_data.values()):
            raise HTTPException(status_code=404, detail="The agent could not generate content for the given topic.")
        return generated_data
    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")