from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import cv2
import io
import base64
import warnings
from typing import List
import os
from typing import List, Optional

from pydantic import BaseModel
from src.model.huggingface_hub_model import huggingface_hub_model
from src.model.open_clip_model import open_clip_model
from src.model.cross_encoder_model import cross_encoder_model
from src.model.open_ai_model import open_ai_model

from dotenv import load_dotenv
load_dotenv()


warnings.filterwarnings("ignore")

app = FastAPI(title="Fashion AI Chatbot", version="3.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def log_requests(request, call_next):
    print(f"[HTTP] Nhận request: {request.method} {request.url.path}")
    response = await call_next(request)
    print(f"[HTTP] Hoàn thành request: {request.method} {request.url.path} -> {response.status_code}")
    return response

os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

device = "cuda" if torch.cuda.is_available() else "cpu"


segment_model = None
siglip_model = None
reranking_model = None
llm = None


all_vectors = None
all_files = None
all_labels = None
data_root = "data"

FASHION_LABELS = {
    1: "hat",
    3: "sunglass", 
    4: "upper-clothes",
    5: "skirt",
    6: "pants",
    7: "dress",
    8: "belt",
    9: "left-shoe",
    10: "right-shoe",
    16: "bag",
    17: "scarf"
}

FASHION_LABELS_VI = {
    1: "Mũ",
    3: "Kính râm", 
    4: "Áo",
    5: "Chân váy",
    6: "Quần",
    7: "Váy liền",
    8: "Thắt lưng",
    9: "Giày trái",
    10: "Giày phải",
    16: "Túi xách",
    17: "Khăn"
}

FASHION_COLORS = {
    1: [255, 0, 0],      # hat - red
    3: [0, 255, 0],      # sunglass - green
    4: [0, 255, 255],    # upper-clothes - cyan
    5: [255, 0, 255],    # skirt - magenta
    6: [128, 0, 128],    # pants - purple
    7: [255, 192, 203],  # dress - pink
    8: [165, 42, 42],    # belt - brown
    9: [255, 165, 0],    # left-shoe - orange
    10: [255, 20, 147],  # right-shoe - deep pink
    16: [75, 0, 130],    # bag - indigo
    17: [255, 105, 180]  # scarf - hot pink
}


class ImageTextQueryRequest(BaseModel):
    text: str
    image_b64: str


class QueryRequest(BaseModel):
    text: str
    conversation_history: List[dict] = [] # Thêm lịch sử trò chuyện

class RefineRequest(BaseModel):
    base_query: str
    feedback_text: Optional[str] = None # Cho phép giá trị là chuỗi hoặc None (null)
    liked_images: List[str] = []     # Định nghĩa rõ ràng là một List (mảng) các chuỗi

@app.on_event("startup")
async def load_model():
    global segment_model
    global siglip_model
    global reranking_model
    global llm
    
    global all_vectors, all_files, all_labels
    global data_root

    # Load segmentation model
    print(f"Loading segmentation model...")
    segment_model = huggingface_hub_model(repo_id="mattmdjaga/segformer_b2_clothes", local_dir="./segformer_b2_clothes")
    print(f"Segment model ready on {device}")

    # Load SigLIP model
    print("Loading SigLIP model...")
    siglip_model = open_clip_model(repo_id="Marqo/marqo-fashionSigLIP", local_dir="./siglip_model", device=device)
    print(f"✅ CLIP model ready on {device}")

    # Load Reranking model
    print("Loading Reranking model...")
    reranking_model = cross_encoder_model('cross-encoder/ms-marco-MiniLM-L-6-v2')
    # check a quick prediction
    test_scores = reranking_model.predict("test query", ["candidate 1", "candidate 2"])
    print(f"Quick test reranking scores: {test_scores}")
    print("Reranking model loaded.")

    # Load LLM model
    print("Loading LLM model...")
    llm = open_ai_model(api_key=os.getenv("OPENAI_API_KEY"), model=os.getenv("OPENAI_API_MODEL", "gpt-4o-mini"), base_url=os.getenv("OPENAI_API_BASE_URL", "https://api.openai.com/v1"))
    print("Quick test LLM...")
    test_response = llm.chat(messages=[{"role": "user", "content": "Hello, are you ready?"}])
    print(f"LLM test response: {test_response}")
    print("LLM model ready.")

    # Load vector dataset
    print("📂 Loading vector dataset...")
    output_root = "vector_database"
    all_vectors_list, all_files_list, all_labels_list = [], [], []
    for npz_file in os.listdir(output_root):
        if npz_file.endswith(".npz"):
            data = np.load(os.path.join(output_root, npz_file), allow_pickle=True)
            all_vectors_list.append(data["vectors"])
            # Lưu trữ toàn bộ đường dẫn tương đối (bao gồm cả label/thư mục)
            all_files_list.extend([os.path.join(data["label"].item(), f) for f in data["filenames"]])
            all_labels_list.extend([data["label"].item()] * len(data["filenames"]))
    all_vectors = np.vstack(all_vectors_list)
    all_vectors = all_vectors / np.linalg.norm(all_vectors, axis=1, keepdims=True)
    all_vectors = all_vectors.astype(np.float32)  # Ensure float32
    all_files = all_files_list
    all_labels = all_labels_list
    print(f"✅ Loaded {len(all_files)} images from dataset")

def extract_fashion_items(image_array, segment_map):
    """Extract individual fashion items as PNG images"""
    fashion_items = []
    
    for label in np.unique(segment_map):
        if label not in FASHION_LABELS:  
            continue
            
        
        mask = (segment_map == label).astype(np.uint8)
        
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)

            
            item_rgba = np.zeros((h, w, 4), dtype=np.uint8)
            
            item_region = image_array[y:y+h, x:x+w]
            item_mask = mask[y:y+h, x:x+w]
            
            # Set RGB channels
            item_rgba[:, :, :3] = item_region
            # Set alpha channel (transparency)
            item_rgba[:, :, 3] = item_mask * 255
            
            # Convert to PIL Image and then to base64
            item_pil = Image.fromarray(item_rgba, 'RGBA')
            buffer = io.BytesIO()
            item_pil.save(buffer, format='PNG')
            item_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            fashion_items.append({
                'label': int(label),
                'name': FASHION_LABELS[label],
                'name_vi': FASHION_LABELS_VI[label],
                'color': FASHION_COLORS[label],
                'bbox': {'x': int(x), 'y': int(y), 'width': int(w), 'height': int(h)},
                'item_png': item_b64,
                'size': {'width': int(w), 'height': int(h)}
            })
    
    return fashion_items

@app.get("/", response_class=HTMLResponse)
async def read_root():
    return FileResponse("src/static/main.html")

@app.post("/analyze")
async def analyze_fashion(file: UploadFile = File(...)):
    try:
        print(f"Analyzing fashion in: {file.filename}")
        
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        image_array = np.array(image)
        
        inputs = segment_model.processor(images=image, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = segment_model.model(**inputs)

        logits = outputs.logits
        
        
        upsampled_logits = F.interpolate(
            logits,
            size=image.size[::-1],  # PIL size is (width, height), need (height, width)
            mode="bilinear",
            align_corners=False,
        )
        
        pred_seg = upsampled_logits.argmax(dim=1)[0].cpu().numpy()
        
        fashion_items = extract_fashion_items(image_array, pred_seg)
        
        orig_buffer = io.BytesIO()
        image.save(orig_buffer, format='PNG')
        original_b64 = base64.b64encode(orig_buffer.getvalue()).decode('utf-8')
        
        print(f"Fashion analysis completed: {len(fashion_items)} items detected")
        
        return {
            "success": True,
            "original_image": original_b64,
            "fashion_items": fashion_items,
            "image_size": {"width": image.width, "height": image.height},
            "total_items": len(fashion_items)
        }
        
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))



# Thay thế endpoint /query_text cũ của bạn bằng cái này

@app.post("/query_text")
async def query_text(request: QueryRequest):
    try:
        # =================================================================
        # GIAI ĐOẠN 0: QUERY REWRITING (Sử dụng LLM nếu có lịch sử trò chuyện)
        # =================================================================
        original_text = request.text
        if not original_text.strip():
            raise HTTPException(status_code=400, detail="Text query cannot be empty")

        final_query = original_text
        if request.conversation_history:
            print(f"[LLM] Rewriting query based on history. Original: '{original_text}'")
            # Tạo prompt cho LLM
            history_str = "\n".join([f"- {msg['role']}: {msg['content']}" for msg in request.conversation_history])
            prompt = f"""Based on the chat history and the user's latest request, generate a new, complete, and concise fashion product search query. Return only the query.

                    Chat history:
                    {history_str}
                    - user: {original_text}

                    New search query:"""
            
            rewritten_query = llm.chat(messages=[{"role": "user", "content": prompt}])
            final_query = rewritten_query.strip()
            print(f"[LLM] Rewritten query: '{final_query}'")

        if len(all_files) == 0:
            raise HTTPException(status_code=500, detail="Dataset not loaded or empty")
        
        print(f"Searching for: '{final_query}'")
        
        # =================================================================
        # GIAI ĐOẠN 1: RETRIEVAL (Lấy ra top 50 ứng viên)
        # =================================================================
        query_emb = siglip_model.text_encoder(final_query)
        sims = all_vectors @ query_emb
        
        # Lấy ra nhiều ứng viên hơn để cho re-ranker xử lý
        candidate_count = 50 
        candidate_indices = sims.argsort()[-candidate_count:][::-1]

        # =================================================================
        # GIAI ĐOẠN 2: RE-RANKING (Chấm điểm lại top 50)
        # =================================================================
        # Tạo các cặp [query, item_description] cho re-ranker
        # Ở đây, ta dùng đường dẫn file làm mô tả, lý tưởng hơn là có metadata sản phẩm
        query = final_query
        candidates = [all_files[idx] for idx in candidate_indices]        
        print(f"Re-ranking {len(candidates)} candidates...")
        rerank_scores = reranking_model.predict(query, candidates)

        # Kết hợp index và score mới lại
        reranked_candidates = list(zip(rerank_scores, candidate_indices))
        # Sắp xếp theo score mới, từ cao đến thấp
        reranked_candidates.sort(key=lambda x: x[0], reverse=True)
        
        # Lấy top 5 kết quả cuối cùng
        topk = 5
        final_indices = [idx for score, idx in reranked_candidates[:topk]]
        
        # =================================================================
        # CHUẨN BỊ KẾT QUẢ TRẢ VỀ
        # =================================================================
        results = []
        for idx in final_indices:
            img_relative_path = all_files[idx]
            img_path = os.path.join(data_root, img_relative_path) 
            try:
                with Image.open(img_path).convert("RGB") as img:
                    buffer = io.BytesIO()
                    img.save(buffer, format='PNG')
                    img_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
                    results.append({
                        "image_b64": img_b64,
                        "label": all_labels[idx],
                        "filename": all_files[idx]
                    })
            except Exception as img_error:
                print(f"Error loading image {img_path}: {img_error}")

        if not results:
            raise HTTPException(status_code=404, detail="No valid images found after re-ranking")
        
        print(f"Query completed: {len(results)} results")
        
        return {
            "success": True,
            "query": final_query, # Trả về query đã được viết lại
            "base_text": original_text,
            "results": results,
            "total_results": len(results)
        }
        
    except Exception as e:
        import traceback
        print(f"Error querying text: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query_image_text")
async def query_image_text(request: ImageTextQueryRequest):
    try:
        text = request.text.strip() if request.text else ""
        image_b64 = request.image_b64

        if not image_b64:
            raise HTTPException(status_code=400, detail="Cần có ảnh cho truy vấn này.")

        print(f"Querying with image and text: '{text}'")

        # Encode image (bắt buộc)
        img_emb = siglip_model.image_encoder(image_b64)
        if img_emb is None:
            raise HTTPException(status_code=400, detail="Ảnh không hợp lệ.")
        
        # Chuẩn hóa vector ảnh
        img_emb /= np.linalg.norm(img_emb)
        query_emb = img_emb # Gán vector đã chuẩn hóa cho query_emb

        # 1. Retrieval (lấy 50 ứng viên)
        candidate_count = 50
        sims = all_vectors @ query_emb
        # Bỏ qua kết quả đầu tiên nếu đó là tìm kiếm ảnh-chính-nó
        candidate_indices = sims.argsort()[-(candidate_count+1):][::-1]
        candidate_indices = candidate_indices[1:] # Bỏ qua top 1

        results = []
        # Nếu không có text, không cần re-rank, trả về kết quả retrieval
        if not text:
            final_indices = candidate_indices[:5]
        else:
            # 2. Re-ranking (dùng text để re-rank)
            query = text
            candidates = [all_files[idx] for idx in candidate_indices]
            print(f"Re-ranking {len(candidates)} candidates for image-text query...")
            rerank_scores = reranking_model.predict(query, candidates)
            reranked_candidates = sorted(zip(rerank_scores, candidate_indices), key=lambda x: x[0], reverse=True)
            final_indices = [idx for score, idx in reranked_candidates[:5]]

        # Chuẩn bị kết quả
        results = []
        for idx in final_indices:
            img_path = os.path.join(data_root, all_files[idx])
            try:
                img = Image.open(img_path).convert("RGB")
                buffer = io.BytesIO()
                img.save(buffer, format='PNG')
                b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
                results.append({
                    "image_b64": b64,
                    "label": all_labels[idx],
                    "filename": os.path.basename(all_files[idx])
                })
            except Exception as img_error:
                print(f"Error loading image {img_path}: {img_error}")

        if not results:
            raise HTTPException(status_code=500, detail="Không tìm thấy ảnh hợp lệ trong kết quả.")

        return {
            "success": True,
            "query": f"[Ảnh] + {text}" if text else "[Ảnh]",
            "base_text": text,
            "results": results,
            "total_results": len(results)
        }

    except Exception as e:
        print(f"Error in image-text query: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))






@app.post("/query_refine")
async def query_refine_advanced(request: RefineRequest):
    try:
        base_query = request.base_query
        feedback_text = request.feedback_text
        liked_images = request.liked_images
        
        # Kiểm tra đầu vào hợp lệ
        if not (feedback_text and feedback_text.strip()) and not liked_images:
            raise HTTPException(status_code=400, detail="Cần cung cấp feedback bằng văn bản hoặc hình ảnh đã thích.")

        # =================================================================
        # GIAI ĐOẠN 0: XÁC ĐỊNH QUERY MỚI DỰA TRÊN LOẠI FEEDBACK
        # =================================================================
        final_query_text = ""
        final_query_emb = None

        # ---- NHÁNH 1: Ưu tiên xử lý feedback bằng văn bản ( tường minh) ----
        if feedback_text and feedback_text.strip():
            print(f"[Refine] Handling text-based feedback. Base: '{base_query}', Feedback: '{feedback_text}'")
            
            prompt = f"""Người dùng đang tìm kiếm sản phẩm thời trang.
            - Truy vấn ban đầu của họ là: "{base_query}"
            - Sau khi xem kết quả, họ đưa ra phản hồi: "{feedback_text}"
            Dựa vào thông tin trên, hãy tạo ra một câu truy vấn tìm kiếm mới, đầy đủ và súc tích hơn, kết hợp cả yêu cầu cũ và mới. Chỉ trả về duy nhất câu truy vấn.
            Câu truy vấn mới:"""

            rewritten_query = llm.chat(messages=[{"role": "user", "content": prompt}])
            final_query_text = rewritten_query.strip()
            final_query_emb = siglip_model.text_encoder(final_query_text)
            print(f"[LLM] Rewritten query for refinement: '{final_query_text}'")

        # ---- NHÁNH 2: Xử lý feedback bằng hình ảnh (ngầm) ----
        elif liked_images:
            print(f"[Refine] Handling image-based feedback. Liked count: {len(liked_images)}")
            
            # Mã hóa query gốc
            base_query_emb = siglip_model.text_encoder(base_query)
            
            # Mã hóa các ảnh đã thích và lấy trung bình
            liked_embs = []
            for img_b64 in liked_images:
                emb = siglip_model.image_encoder(img_b64)
                if emb is not None:
                    liked_embs.append(emb)
            
            if not liked_embs:
                raise HTTPException(status_code=400, detail="Không có ảnh hợp lệ nào trong danh sách đã thích.")

            avg_liked_emb = np.mean(np.array(liked_embs), axis=0)

            # Tạo query embedding mới bằng thuật toán Rocchio
            # Trọng số: 70% từ query gốc, 30% từ ảnh đã thích
            alpha = 0.7
            beta = 0.3
            final_query_emb = alpha * base_query_emb + beta * avg_liked_emb
            final_query_emb /= np.linalg.norm(final_query_emb) # Chuẩn hóa lại vector

            # QUAN TRỌNG: Với re-ranking, ta vẫn dùng text của query GỐC
            # vì đó là ý định rõ ràng nhất của người dùng bằng ngôn ngữ.
            final_query_text = base_query
            print(f"[Refine] Created new embedding from liked images. Using base query '{base_query}' for re-ranking.")


        # =================================================================
        # GIAI ĐOẠN 1 & 2: RETRIEVAL + RE-RANKING (Dùng chung cho cả 2 nhánh)
        # =================================================================
        
        # 1. Retrieval (dùng final_query_emb đã được tính toán)
        sims = all_vectors @ final_query_emb
        candidate_count = 50
        candidate_indices = sims.argsort()[-candidate_count:][::-1]

        # 2. Re-ranking (dùng final_query_text)
        query = final_query_text
        candidates = [all_files[idx] for idx in candidate_indices]
        print(f"Re-ranking {len(candidates)} candidates using text: '{final_query_text}'")
        rerank_scores = reranking_model.predict(query, candidates)

        reranked_candidates = sorted(zip(rerank_scores, candidate_indices), key=lambda x: x[0], reverse=True)
        
        topk = 5
        final_indices = [idx for score, idx in reranked_candidates[:topk]]
        
        # ... (Phần chuẩn bị kết quả giữ nguyên, không cần thay đổi) ...
        results = []
        for idx in final_indices:
            img_relative_path = all_files[idx]
            img_path = os.path.join(data_root, img_relative_path)
            try:
                with Image.open(img_path).convert("RGB") as img:
                    buffer = io.BytesIO()
                    img.save(buffer, format='PNG')
                    img_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
                    results.append({
                        "image_b64": img_b64,
                        "label": all_labels[idx],
                        "filename": all_files[idx]
                    })
            except Exception as img_error:
                print(f"[Refine] Error loading image {img_path}: {img_error}")

        if not results:
            raise HTTPException(status_code=404, detail="No valid images found for the refined query.")

        print(f"[Refine] Completed with {len(results)} results.")
        
        display_query = final_query_text if (feedback_text and feedback_text.strip()) else f"{base_query} (tinh chỉnh theo ảnh đã thích)"
        
        return {
            "success": True,
            "query": display_query,
            "base_text": feedback_text if feedback_text else "image_feedback",
            "results": results,
            "total_results": len(results)
        }

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        print(f"[Refine] Unexpected error: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)