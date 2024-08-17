# relevant libraies
import os
import openai
import faiss
import numpy as np
from dotenv import load_dotenv
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer


# Load environment variables from .env file
load_dotenv('../.env')

# Load OpenAI API key
api_key = os.getenv("OPENAI_API_KEY")
if api_key is None:
    raise ValueError("OpenAI API key is not set in the environment variables.")
openai.api_key = api_key


import pickle

# Load vector store from disk
with open('../data/vector_stored.pkl', 'rb') as f:
    vector_store = pickle.load(f)



# Load your SentenceTransformer model
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

embeddings_np = np.array(vector_store['embeddings'])
pca = PCA(n_components=100)  # Adjust the number of components as needed
embeddings_reduced = pca.fit_transform(embeddings_np)
vector_store['embeddings_reduced'] = embeddings_reduced

d = embeddings_reduced.shape[1]  # Dimensionality of the reduced embeddings
index = faiss.IndexFlatL2(d)  # L2 distance for exact search
index.add(embeddings_reduced)  # Add the reduced embeddings to the index


class QAChatAgent:
    def __init__(self, model, pca, index, vector_store, top_k=5):
        self.model = model
        self.pca = pca
        self.index = index
        self.vector_store = vector_store
        self.top_k = top_k
        self.conversation_history = []  # To store the conversation history

    def add_to_history(self, query, answer):
        self.conversation_history.append({"query": query, "answer": answer})

    def generate_answer_gpt4(self, relevant_documents, question):
        context = "\n\n".join([f"User: {entry['query']}\nAssistant: {entry['answer']}" for entry in self.conversation_history])
        context += "\n\n" + "\n\n".join(relevant_documents)
        prompt = f"Based on the following documents and conversation history, answer the question:\n\n{context}\n\nQuestion: {question}\nAnswer:"

        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=350,
            temperature=0.0
        )
        
        return response.choices[0].message['content'].strip()

    def search_vector_store(self, query):
        query_embedding = self.model.encode(query)
        query_embedding_reduced = self.pca.transform([query_embedding]).astype('float32')
        D, I = self.index.search(query_embedding_reduced, self.top_k)
        results = [(self.vector_store['documents'][i], self.vector_store['metadatas'][i], D[0][n]) for n, i in enumerate(I[0])]
        return results

    def process_query(self, query):
        if query.lower().strip() == "quit":
            print("Ending conversation. Goodbye!")
            return None
        
        results = self.search_vector_store(query)
        relevant_documents = [doc for doc, _, _ in results]
        answer = self.generate_answer_gpt4(relevant_documents, query)
        
        # Add the query and answer to the conversation history
        self.add_to_history(query, answer)
        
        return answer
    

# Usage
qa_agent = QAChatAgent(model, pca, index, vector_store)

# Simulate conversation loop
while True:
    user_query = input("You: ")
    response = qa_agent.process_query(user_query)
    
    if response is None:
        break  # Exit the loop if "quit" is typed
    
    print(f"rgu_bot: {response}")


