from typing import List
import numpy as np
from src.utils.config import topk

def retrieval(all_vectors: np.ndarray, query_emb: np.ndarray):
    print("Current topk:", topk)
    sims = all_vectors @ query_emb
    candidate_indices = sims.argsort()[-topk:][::-1]
    return candidate_indices, sims[candidate_indices]