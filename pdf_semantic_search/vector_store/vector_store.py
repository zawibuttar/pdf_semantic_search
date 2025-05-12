from pinecone import Pinecone, ServerlessSpec
from config import PINECONE_API_KEY, PINECONE_INDEX_NAME

class PineconeVectorStore:
    def __init__(self):
        self.pc = Pinecone(api_key=PINECONE_API_KEY)
        if PINECONE_INDEX_NAME not in self.pc.list_indexes().names():
            self.pc.create_index(
                name=PINECONE_INDEX_NAME,
                dimension=1024,
                metric='cosine',
                spec=ServerlessSpec(cloud='aws', region='us-east-1')
            )
        self.index = self.pc.Index(PINECONE_INDEX_NAME)

    def upsert(self, texts, embeddings):
        vectors = [
            (f"id-{i}", emb.tolist(), {"text": text})
            for i, (text, emb) in enumerate(zip(texts, embeddings))
        ]
        self.index.upsert(vectors)

    def search(self, query_embedding, top_k=3):
        results = self.index.query(vector=query_embedding.tolist(), top_k=top_k, include_metadata=True)
        return [match['metadata']['text'] for match in results['matches']]
