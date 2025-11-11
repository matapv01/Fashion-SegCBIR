from typing import List
import io
import base64
from src.utils.config import topk, topn, threshold
from src.utils.load_image import load_image_from_path_or_url


def rerank(reranking_model, query_text: str, all_files: List[str], all_labels: List[str], candidate_indices: List[int], is_rerank: bool = True, eval_mode: bool = False):
    if is_rerank:
        print("Reranking with threshold:", threshold)
        print("Reranking to topn:", topn)
        scores = reranking_model.predict(query_text, [all_files[idx] for idx in candidate_indices])
        reranked = sorted(zip(scores, candidate_indices), key=lambda x: x[0], reverse=True)

        if eval_mode:
            filtered = [(s, idx) for s, idx in reranked]
        else:
            filtered = [(s, idx) for s, idx in reranked if s >= threshold]
            
        final_indices = [idx for _, idx in filtered[:topn]]
        # print(f"""Scores after reranking: {[score for score, _ in filtered[:topn]]}""")

        results = []
        for idx in final_indices:
            try:
                with load_image_from_path_or_url(all_files[idx]) as img:
                    buffer = io.BytesIO()
                    img.save(buffer, format='PNG')
                    results.append({
                        "image_b64": base64.b64encode(buffer.getvalue()).decode("utf-8"),
                        "label": all_labels[idx],
                        "filename": all_files[idx]
                    })
            except Exception as e:
                print(f"Error loading image {all_files[idx]}: {e}")
        return results
    else:
        print("Reranking skipped. Select topk results directly.")
        final_indices = candidate_indices[:topn]

        if eval_mode:
            final_indices = candidate_indices
        else:
            final_indices = [idx for idx in candidate_indices[:topn] if idx >= threshold]

        results = []
        for idx in final_indices:
            try:
                with load_image_from_path_or_url(all_files[idx]) as img:
                    buffer = io.BytesIO()
                    img.save(buffer, format='PNG')
                    results.append({
                        "image_b64": base64.b64encode(buffer.getvalue()).decode("utf-8"),
                        "label": all_labels[idx],
                        "filename": all_files[idx]
                    })
            except Exception as e:
                print(f"Error loading image {all_files[idx]}: {e}")
        return results