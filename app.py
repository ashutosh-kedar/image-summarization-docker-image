from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

from inference import predict

app = FastAPI(title="Qwen2-VL SageMaker")


class InferenceRequest(BaseModel):
    prompt: str
    images: List[str]
    max_new_tokens: int = 512
    temperature: float = 0.2
    do_sample: bool = False


@app.get("/ping")
def ping():
    return {"status": "healthy"}


@app.post("/invocations")
def invocations(request: InferenceRequest):

    try:
        return predict(request.model_dump())

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e),
        )