import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# Initialize the SentenceTransformer model
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

def load_vector_store(filepath):
    """Load the vector store from a JSON file."""
    with open(filepath, 'r') as file:
        vector_store = json.load(file)
    return vector_store

def generate_query_embedding(query):
    """Generate embedding for a given query."""
    return model.encode(query)

def find_similar(query, vector_store, top_n=5):
    """Find the most similar documents to the query based on cosine similarity."""
    try:
        if not all(key in vector_store for key in ['documents', 'embeddings', 'metadatas']):
            raise ValueError("Vector store must contain 'documents', 'embeddings', and 'metadatas' keys.")

        query_embedding = generate_query_embedding(query)
        assert len(query_embedding) == len(vector_store['embeddings'][0]), "Embedding dimensions do not match."

        similarities = cosine_similarity([query_embedding], vector_store['embeddings'])[0]
        similar_indices = np.argsort(similarities)[-top_n:][::-1]

        similar_docs = [vector_store['documents'][i] for i in similar_indices]
        
        return similar_docs
    
    except Exception as e:
        print(f"An error occurred: {e}")
        return []

