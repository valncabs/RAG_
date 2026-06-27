from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.rag_chain import (
    initialize_chain,
    get_answer,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    initialize_chain()
    yield


app = FastAPI(
    title="RAG API",
    description="Sistema RAG con LangChain + Chroma + Groq",
    version="1.0.0",
    lifespan=lifespan,
)

# ==========================
# CORS
# ==========================
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:4200",
        "http://127.0.0.1:4200",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================

class QuestionRequest(BaseModel):
    question: str


@app.get("/")
def home():
    return {
        "message": "RAG API funcionando correctamente"
    }


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return Response(status_code=204)


@app.post("/ask")
def ask(request: QuestionRequest):
    try:
        result = get_answer(request.question)
        return result

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )