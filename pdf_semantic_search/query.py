from Emedder.embedder import TextEmbedder
from vector_store.vector_store import PineconeVectorStore
from config import *
def search_pdf(query: str, top_k=3):
    embedder = TextEmbedder(EMBEDDING_MODEL)
    embeddings = embedder.embed_texts(query)
    vector_store = PineconeVectorStore()

    query_embedding = embeddings
    results = vector_store.search(query_embedding, top_k=top_k)
    return results

if __name__ == "__main__":
    print("\n📄 Semantic PDF Search (type 'exit' to quit)")
    while True:
        query = input("\n🔍 Enter your query: ")
        if query.lower() in ["exit", "quit"]:
            break
        print("\n🔎 Searching...\n")
        results = search_pdf(query)
        for i, res in enumerate(results, 1):
            print(f"Result {i}:\n{res}\n{'-'*60}")
