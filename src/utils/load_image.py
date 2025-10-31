import requests
from PIL import Image
import io
import os
from src.utils.config import data_root

def load_image_from_path_or_url(path: str):
    """Load ảnh từ local hoặc URL"""
    if path.startswith("http://") or path.startswith("https://"):
        response = requests.get(path, timeout=10)
        response.raise_for_status()
        return Image.open(io.BytesIO(response.content)).convert("RGB")
    else:
        return Image.open(os.path.join(data_root, path)).convert("RGB")