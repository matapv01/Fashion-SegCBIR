import torch
import numpy as np
import os
import io
import json
import base64
import requests
import warnings
from PIL import Image
from datetime import datetime
from collections import Counter, defaultdict
from dotenv import load_dotenv


from src.utils.load_metadata import load_metadata
from src.utils.load_models import load_models
from src.utils.load_image import load_image_from_path_or_url

warnings.filterwarnings("ignore")
load_dotenv()

device = "cuda" if torch.cuda.is_available() else "cpu"


# GLOBAL VARIABLES

segment_model = None
siglip_model = None
reranking_model = None
llm = None

all_vectors = None
all_files = None
all_labels = None

data_root = "data"
topk = 5



def save_prediction_record(query, indices, scores, save_path="predictions_all.json"):
    """Lưu kết quả của từng subcategory vào JSON"""
    global all_files, all_labels

    labels = [all_labels[idx] for idx in indices]
    main_label = Counter(labels).most_common(1)[0][0] if labels else None

    record = {
        "timestamp": datetime.now().isoformat(),
        "query": query,
        "subcategory": main_label,
        "results": [
            {
                "filename": all_files[idx],
                "label": all_labels[idx],
                "score": float(scores[i])
            }
            for i, idx in enumerate(indices)
        ]
    }

    if os.path.exists(save_path):
        with open(save_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = []

    data.append(record)

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"✅ Saved {query} -> {save_path}")



# QUERY PIPELINE


def run_query(text_query: str):
    """Chạy retrieval cho một query text"""
    global siglip_model, reranking_model, all_vectors, all_files, all_labels

    if not text_query.strip():
        print("❌ Empty query.")
        return None

    print(f"\n🔍 Query: {text_query}")

    # 1️⃣ Encode query
    query_emb = siglip_model.text_encoder(text_query)
    sims = all_vectors @ query_emb

    # 2️⃣ Lấy top N ứng viên ban đầu
    candidate_count = 50
    candidate_indices = sims.argsort()[-candidate_count:][::-1]
    candidates = [all_files[idx] for idx in candidate_indices]

    # 3️⃣ Rerank
    rerank_scores = reranking_model.predict(text_query, candidates)
    reranked = list(zip(rerank_scores, candidate_indices))
    reranked.sort(key=lambda x: x[0], reverse=True)

    filtered = [(score, idx) for score, idx in reranked]
 

    final_indices = [idx for score, idx in filtered[:topk]]
    final_scores = [score for score, idx in filtered[:topk]]

    # 4️⃣ Hiển thị top kết quả
    print(f"✅ Top {len(final_indices)} results:")
    for i, (score, idx) in enumerate(zip(final_scores, final_indices)):
        print(f"  {i+1}. {all_files[idx]} | label={all_labels[idx]} | score={score:.4f}")

    return final_indices


def run_predict():
    """Chạy đánh giá cho tất cả subcategory trong dataset"""
    global segment_model, siglip_model, reranking_model, llm
    global all_vectors, all_files, all_labels, device

    segment_model, siglip_model, reranking_model, llm = load_models(segment_model, siglip_model, reranking_model, llm, device=device)
    _, all_vectors, all_files, all_labels = load_metadata()
    print("\n🧠 Running queries for all subcategories...")
    unique_labels = sorted(list(set(all_labels)))
    print(f"Found {len(unique_labels)} unique labels:")
    print(unique_labels)

    all_predictions = {}

    for label in unique_labels:
        print(f"\n==============================")
        print(f"🎯 Querying for subcategory: {label}")

        # define query for label
        query = f"""A high-quality photo of a person wearing {label}"""

        indices = run_query(query)
        if indices:
            # Lưu kết quả trong dict
            all_predictions[label] = {
                "results": [
                    {
                        "filename": all_files[idx],
                        "label": all_labels[idx]
                    }
                    for idx in indices
                ]
            }
           

    # Cuối cùng mới ghi ra file 1 lần
    with open("logs/predictions_all.json", "w", encoding="utf-8") as f:
        json.dump(all_predictions, f, ensure_ascii=False, indent=2)


    print("\n🎯 All subcategory queries completed!")
    print(f"Saved per-query results in logs/predictions_all.json")