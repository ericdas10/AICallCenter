import chromadb
from sentence_transformers import SentenceTransformer
import os
import uuid

class Embedder:
    def __init__(self, persist_directory="data/vector_store"):
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(name="documents")
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def add_document(self, text, metadata=None):
        embedding = self.model.encode([text])[0]
        doc_id = str(uuid.uuid4())
        self.collection.add(
            ids=[doc_id],
            embeddings=[embedding.tolist()],
            metadatas=[metadata or {}],
            documents=[text]
        )

    def search(self, query, top_k=3):
        query_embedding = self.model.encode([query])[0]
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k
        )
        return results