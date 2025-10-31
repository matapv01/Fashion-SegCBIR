from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import io
import base64
import warnings
from typing import List, Optional
from src.utils.FashionAI import FashionAI
from src.utils.load_models import load_models
from src.utils.load_image import load_image_from_path_or_url

from src.node.text_encoder import text_encoder
from src.node.image_encoder import image_encoder
from src.node.retrieval import retrieval
from src.node.rerank import rerank
from src.node.rewrite_query import rewrite_query
from src.node.refine_query_text import refine_query_text
from src.node.refine_query_image import refine_query_image

from src.utils.config import *
from dotenv import load_dotenv



load_dotenv()
warnings.filterwarnings("ignore")

app = FastAPI(title="Fashion AI Chatbot", version="3.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/home", StaticFiles(directory="src/static"), name="static")

# ------------------ Pydantic Models ------------------
class ImageTextQueryRequest(BaseModel):
    text: str
    image_b64: str

class QueryRequest(BaseModel):
    text: str
    conversation_history: List[dict] = []

class RefineRequest(BaseModel):
    base_query: str
    feedback_text: Optional[str] = None
    liked_images: List[str] = []

    


# ------------------ Initialize ------------------
fashion_ai = None
segment_model, siglip_model, reranking_model, llm = None, None, None, None

@app.on_event("startup")
async def startup_event():
    global fashion_ai, segment_model, siglip_model, reranking_model, llm
    segment_model, siglip_model, reranking_model, llm = load_models()
    fashion_ai = FashionAI()



# ------------------ Endpoints ------------------
@app.get("/", response_class=HTMLResponse)
async def root():
    return FileResponse("src/static/main.html")

@app.post("/query_text")
async def query_text(request: QueryRequest):
    query = rewrite_query(llm, request.text, request.conversation_history)
    emb = text_encoder(siglip_model, query)
    candidate_indices, _ = retrieval(fashion_ai.all_vectors, emb)
    results = rerank(reranking_model, query, fashion_ai.all_files, fashion_ai.all_labels, candidate_indices)
    return {"success": True, "query": query, "base_text": request.text, "results": results, "total_results": len(results)}

@app.post("/query_image_text")
async def query_image_text(request: ImageTextQueryRequest):
    text = request.text.strip()
    text_emb = text_encoder(siglip_model, text)
    img_emb = image_encoder(siglip_model, request.image_b64)
    query_emb = 0.7*text_emb + 0.3*img_emb
    query_emb /= np.linalg.norm(query_emb)
    candidate_indices, _ = retrieval(fashion_ai.all_vectors, query_emb)
    results = rerank(reranking_model, text if text else "[Ảnh]", fashion_ai.all_files, fashion_ai.all_labels, candidate_indices)
    return {"success": True, "query": f"[Ảnh] + {text}" if text else "[Ảnh]",
            "base_text": text, "results": results, "total_results": len(results)}

@app.post("/query_refine")
async def query_refine(request: RefineRequest):
    if request.feedback_text and request.feedback_text.strip():
        query = refine_query_text(llm, request.base_query, request.feedback_text)
        emb = text_encoder(siglip_model, query)
        candidate_indices, _ = retrieval(fashion_ai.all_vectors, emb)
        results = rerank(reranking_model, query, fashion_ai.all_files, fashion_ai.all_labels, candidate_indices)
    elif request.liked_images:
        emb = refine_query_image(siglip_model, request.base_query, request.liked_images)
        candidate_indices, _ = retrieval(fashion_ai.all_vectors, emb)
        results = rerank(reranking_model, request.base_query, fashion_ai.all_files, fashion_ai.all_labels, candidate_indices)
        query = request.base_query
    else:
        raise HTTPException(status_code=400, detail="Provide feedback text or liked images")
    return {"success": True, "query": query, "base_text": request.feedback_text or "image_feedback",
            "results": results, "total_results": len(results)}

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
