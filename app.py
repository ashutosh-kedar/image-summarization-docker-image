from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from inference import predict, load_model


@asynccontextmanager
async def lifespan(app):
    load_model()  # download + load before /ping reports healthy
    yield


app = FastAPI(title="Qwen2-VL SageMaker API", lifespan=lifespan)


class ChatTurn(BaseModel):
    role: str  # "user" | "assistant" | "system"
    content: str


class InferenceRequest(BaseModel):
    prompt: str
    user_id: Optional[str] = None
    image_names: List[str] = []
    chat_history: List[ChatTurn] = []
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
    except (ValueError, FileNotFoundError) as e:
        # Bad input (missing image, wrong extension, missing user_id)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)