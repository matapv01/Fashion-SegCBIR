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
from pydantic import BaseModel
from src.model.huggingface_hub_model import huggingface_hub_model
from src.model.open_clip_model import open_clip_model

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

class QueryRequest(BaseModel):
    text: str

class ImageTextQueryRequest(BaseModel):
    text: str
    image_b64: str

class RefineQueryRequest(BaseModel):
    text: str
    liked_images: List[str] # Vẫn giữ nguyên kiểu List[str] cho đến khi nhận được dữ liệu từ client

@app.on_event("startup")
async def load_model():
    global segment_model
    global siglip_model
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




@app.post("/query_text")
async def query_text(request: QueryRequest):
    try:
        text = request.text
        if not text.strip():
            raise HTTPException(status_code=400, detail="Text query cannot be empty")
        
        if len(all_files) == 0:
            raise HTTPException(status_code=500, detail="Dataset not loaded or empty")
        
        print(f"Querying text: {text}")
        
        # Encode text y hệt test_retrival.py
        query_emb = siglip_model.text_encoder(text)

        # Cosine similarity
        sims = all_vectors @ query_emb
        topk = 5
        top_idx = sims.argsort()[-topk:][::-1]
        
        # Prepare results
        results = []
        for idx in top_idx:
            # all_files đã chứa đường dẫn tương đối bao gồm cả label, ví dụ: "hat/image.jpg"
            # Cần kết hợp với data_root để có đường dẫn tuyệt đối
            img_relative_path = all_files[idx] # Ví dụ: "hat/image_0001.jpg"
            img_path = os.path.join(data_root, img_relative_path) 
            try:
                img = Image.open(img_path).convert("RGB")
                buffer = io.BytesIO()
                img.save(buffer, format='PNG')
                img_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
                results.append({
                    "image_b64": img_b64,
                    "label": all_labels[idx],
                    "filename": all_files[idx] # Lưu trữ đường dẫn tương đối để dễ dàng tìm kiếm lại
                })
            except Exception as img_error:
                print(f"Error loading image {img_path}: {img_error}")
                # Skip invalid images
        
        if not results:
            raise HTTPException(status_code=500, detail="No valid images found in dataset")
        
        print(f"Text query completed: {len(results)} results")
        
        return {
            "success": True,
            "query": text,
            "base_text": text,
            "results": results,
            "total_results": len(results)
        }
        
    except Exception as e:
        print(f"Error querying text: {str(e)}")
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

        if text:  # Có text
            text_emb = siglip_model.text_encoder(text)
            query_emb = text_emb * 0.6 + img_emb * 0.4
        else:     # Không có text, chỉ dùng ảnh
            query_emb = img_emb

        # Chuẩn hóa vector
        query_emb /= np.linalg.norm(query_emb)

        # Cosine similarity
        sims = all_vectors @ query_emb
        topk = 5
        top_idx = sims.argsort()[-(topk+1):][::-1]
        top_idx = top_idx[1: topk+1]  # Bỏ ảnh đầu tiên (ảnh gốc)

        # Chuẩn bị kết quả
        results = []
        for idx in top_idx:
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
async def query_refine(request: RefineQueryRequest):
    try:
        text = request.text
        liked_images = request.liked_images
        if not text.strip() or not liked_images:
            raise HTTPException(status_code=400, detail="Invalid refine request")

        print(f"[Refine] Received request → text='{text}', liked_count={len(liked_images)}")
        if liked_images:
            print(f"[Refine] First liked image base64 length: {len(liked_images[0])}")

        text_emb = siglip_model.text_encoder(text)

        img_embs = []
        for idx, img_b64 in enumerate(liked_images, start=1):
            print(f"[Refine] Encoding liked image #{idx}, base64 length={len(img_b64)}")
            try:
                emb = siglip_model.image_encoder(img_b64)
                if emb is not None:
                    img_embs.append(emb)
                    print(f"[Refine] ✅ Encoded liked image #{idx}")
                else:
                    print(f"[Refine] ⚠️ Failed to encode liked image #{idx}")
            except Exception as encode_err:
                print(f"[Refine] ⚠️ Exception encoding liked image #{idx}: {encode_err}")

        if not img_embs:
            print("[Refine] No valid liked images after encoding.")
            raise HTTPException(status_code=400, detail="No valid liked images")

        img_embs = np.stack(img_embs, axis=0)
        img_emb = np.mean(img_embs, axis=0)
        img_emb /= np.linalg.norm(img_emb)

        query_emb = text_emb * 0.6 + img_emb * 0.4
        query_emb /= np.linalg.norm(query_emb)
        print(f"[Refine] Combined embedding ready. Norm={np.linalg.norm(query_emb):.4f}")

        sims = all_vectors @ query_emb
        topk = 5
        top_idx = sims.argsort()[-topk:][::-1]

        results = []
        for idx in top_idx:
            img_path = os.path.join(data_root, str(all_labels[idx]), os.path.basename(all_files[idx]))
            try:
                img = Image.open(img_path).convert("RGB")
                buffer = io.BytesIO()
                img.save(buffer, format='PNG')
                img_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
                results.append({
                    "image_b64": img_b64,
                    "label": all_labels[idx],
                    "filename": os.path.basename(all_files[idx])
                })
            except Exception as img_error:
                print(f"[Refine] Error loading image {img_path}: {img_error}")

        if not results:
            print("[Refine] No valid images found for result.")
            raise HTTPException(status_code=500, detail="No valid images found")

        print(f"[Refine] Completed with {len(results)} results.")
        return {
            "success": True,
            "query": f"Refined: {text} + {len(liked_images)} likes",
            "base_text": text,
            "results": results,
            "total_results": len(results)
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"[Refine] Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)