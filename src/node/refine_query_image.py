from src.node.text_encoder import text_encoder
from src.node.image_encoder import image_encoder
from src.utils.config import text_weight, image_weight

from typing import List
import numpy as np



def refine_query_image(siglip_model, base_query: str, liked_images: List[str]) -> np.ndarray:
    base_emb = text_encoder(siglip_model, base_query)
    liked_embs = [image_encoder(siglip_model, img) for img in liked_images if image_encoder(siglip_model, img) is not None]
    avg_emb = np.mean(np.array(liked_embs), axis=0)
    final_emb = text_weight*base_emb + image_weight*avg_emb
    final_emb /= np.linalg.norm(final_emb)
    return final_emb