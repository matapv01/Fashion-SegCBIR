from typing import List
import numpy as np

def text_encoder(siglip_model, text: str) -> np.ndarray:
    return siglip_model.text_encoder(text)