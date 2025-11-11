
from src.model.huggingface_hub_model import huggingface_hub_model
from src.model.open_clip_model import open_clip_model
from src.model.cross_encoder_model import cross_encoder_model
from src.model.open_ai_model import open_ai_model
import os
from src.utils.config import device

def load_models(is_reranking_model: bool = True):

    # Load segmentation model
    print(f"Loading segmentation model...")
    segment_model = huggingface_hub_model(repo_id="mattmdjaga/segformer_b2_clothes", local_dir="./segformer_b2_clothes", device=device)
    print(f"Segment model ready on {device}")

    # Load SigLIP model
    print("Loading SigLIP model...")
    siglip_model = open_clip_model(repo_id="Marqo/marqo-fashionSigLIP", local_dir="./siglip_model", device=device)
    print(f"✅ CLIP model ready on {device}")

    if is_reranking_model:
        # Load Reranking model
        print("Loading Reranking model...")
        reranking_model = cross_encoder_model('cross-encoder/ms-marco-MiniLM-L-6-v2', device=device)
        # check a quick prediction
        test_scores = reranking_model.predict("test query", ["candidate 1", "candidate 2"])
        print(f"Quick test reranking scores: {test_scores}")
        print("Reranking model loaded.")
    else:
        reranking_model = None

    # Load LLM model
    print("Loading LLM model...")
    llm = open_ai_model(api_key=os.getenv("OPENAI_API_KEY"), model=os.getenv("OPENAI_API_MODEL", "gpt-4o-mini"), base_url=os.getenv("OPENAI_API_BASE_URL", "https://api.openai.com/v1"))
    # print("Quick test LLM...")
    # test_response = llm.chat(messages=[{"role": "user", "content": "Hello, are you ready?"}])
    # print(f"LLM test response: {test_response}")
    print("LLM model ready.")

    return segment_model, siglip_model, reranking_model, llm