from sentence_transformers.cross_encoder import CrossEncoder

class cross_encoder_model:
    def __init__(self, model_name: str, device: str = "cpu"):
        self.model = CrossEncoder(model_name)
        self.model.to(device)

    def predict(self, query: str, candidates: list):
        return self.model.predict([(query, candidate) for candidate in candidates])
