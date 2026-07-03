from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from contextlib import asynccontextmanager
from inference import predict, load_model

@asynccontextmanager
async def lifespan(app):
    load_model()   # download + load before serving traffic
    yield

app = FastAPI(title="Qwen2-VL SageMaker API", lifespan=lifespan)


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
        raise HTTPException(status_code=500, detail=str(e))


# ✅ IMPORTANT: REQUIRED for SageMaker container startup
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)