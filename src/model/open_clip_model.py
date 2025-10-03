
import base64
from PIL import Image
from huggingface_hub import snapshot_download
import os

import io
import torch
import open_clip

class open_clip_model:
    def __init__(self, repo_id, local_dir, device):
        self.repo_id = repo_id
        self.local_dir = local_dir
        self.device = device
        if not os.path.exists(local_dir):
            self.load_model()

        self.model, _, self.processor = open_clip.create_model_and_transforms(
            f"hf-hub:{self.repo_id}",
            cache_dir="siglip_model"
        )
        self.model = self.model.to(device, dtype=torch.float16)

        self.tokenizer = open_clip.get_tokenizer(f"hf-hub:{repo_id}")


    def load_model(self):
        os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
        snapshot_download(
            repo_id=self.repo_id,
            local_dir=self.local_dir,   
            local_dir_use_symlinks=False          
        )
    

    def text_encoder(self, query_text: str):
        text_inputs = self.tokenizer([query_text]).to(self.device)
        with torch.no_grad():
            emb = self.model.encode_text(text_inputs)
        emb /= emb.norm(dim=-1, keepdim=True)
        return emb.squeeze().cpu().numpy()
    

    def image_encoder(self, image_b64: str):
        image_data = base64.b64decode(image_b64)
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        image_input = self.processor(image).unsqueeze(0).to(self.device, dtype=torch.float16)
        with torch.no_grad():
            emb = self.model.encode_image(image_input)
        emb /= emb.norm(dim=-1, keepdim=True)
        return emb.squeeze().cpu().numpy()

