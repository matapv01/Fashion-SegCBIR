
import os
from huggingface_hub import snapshot_download
from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation

class huggingface_hub_model:
    def __init__(self, repo_id: str, local_dir: str = "./model", device: str = "cpu"):
        self.repo_id = repo_id
        self.local_dir = local_dir
        self.device = device

        if not os.path.exists(local_dir):
            self.load_model()
        
        self.model = AutoModelForSemanticSegmentation.from_pretrained(repo_id)
        self.processor = SegformerImageProcessor.from_pretrained(repo_id)
        self.model.to(device)
        self.model.eval()

    def load_model(self):
        os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
        snapshot_download(
            repo_id=self.repo_id,
            local_dir=self.local_dir,   
            local_dir_use_symlinks=False          
        )

