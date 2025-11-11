import os
import json
import warnings
import sys
from datetime import datetime
from collections import Counter
from dotenv import load_dotenv



from src.utils.load_models import load_models
from src.utils.FashionAI import FashionAI
from src.utils.config import *
from src.node.retrieval import retrieval
from src.node.rerank import rerank
from src.node.text_encoder import text_encoder

warnings.filterwarnings("ignore")
load_dotenv()



# GLOBAL VARIABLES

segment_model, siglip_model, reranking_model, llm = load_models(is_reranking_model=False)


fashion_ai = FashionAI()
all_vectors = fashion_ai.all_vectors
all_files = fashion_ai.all_files
all_labels = fashion_ai.all_labels




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
    """Chạy retrieval + rerank cho một query text, dùng các hàm đã định nghĩa"""
    global siglip_model, reranking_model, all_vectors, all_files, all_labels

    if not text_query.strip():
        print("❌ Empty query.")
        return None

    print(f"\n🔍 Query: {text_query}")

    # 1️⃣ Encode query
    query_emb = text_encoder(siglip_model, text_query)  # node_encode_text

    # 2️⃣ Retrieval: lấy candidate_indices & similarity
    candidate_indices, _ = retrieval(all_vectors, query_emb)  # node_retrieval

    # 3️⃣ Rerank: dùng text_query để re-rank top candidates
    results = rerank(reranking_model, text_query, all_files, all_labels, candidate_indices, is_rerank=False, eval_mode=True)  # node_rerank eval mode

    # 4️⃣ Hiển thị top kết quả
    print(f"✅ Top {len(results)} results:")
    for i, res in enumerate(results):
        print(f"  {i+1}. {res['filename']} | label={res['label']}")

    # Trả về danh sách final indices
    final_indices = [all_files.index(res["filename"]) for res in results]
    return final_indices



def run_predict():
    """Chạy đánh giá cho tất cả subcategory trong dataset"""
    global segment_model, siglip_model, reranking_model, llm
    global all_vectors, all_files, all_labels, device
    
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