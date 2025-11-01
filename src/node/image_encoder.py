import numpy as np


def image_encoder(siglip_model, image_b64: str) -> np.ndarray:
    return siglip_model.image_encoder(image_b64)