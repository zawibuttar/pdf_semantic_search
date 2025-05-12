from sentence_transformers import SentenceTransformer

class TextEmbedder:
    def __init__(self, model_name):
        self.model = SentenceTransformer(model_name)

    def embed_texts(self, texts):
        return self.model.encode(texts)
