from config import EMBEDDING_MODEL
from utils.utils import read_pdf, chunk_text
from Emedder.embedder import TextEmbedder
from vector_store.vector_store import PineconeVectorStore
import os
def main():
    input_dir = "input"

    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"The directory '{input_dir}' does not exist.")
    else:
        print(f"The directory '{input_dir}' exists.")
    
    texts = read_pdf(f"{input_dir}/sample.pdf")
    chunks = chunk_text(texts)
    embedder = TextEmbedder(EMBEDDING_MODEL)
    embeddings = embedder.embed_texts(chunks)

    # Upload to Pinecone
    vector_store = PineconeVectorStore()
    vector_store.upsert(chunks, embeddings)

if __name__ == "__main__":
    main()
