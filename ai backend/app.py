from fastapi import FastAPI
from pydantic import BaseModel
from model_logic import get_response
import uvicorn

app = FastAPI()

class ChatRequest(BaseModel):
    user_input: str

@app.post("/chat/")
async def chat(request: ChatRequest):
    response = get_response(request.user_input)
    return {"response": response}

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
