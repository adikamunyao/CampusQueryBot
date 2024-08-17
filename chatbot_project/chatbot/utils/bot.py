import os
import openai
import faiss
import numpy as np
from dotenv import load_dotenv
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer
import pickle
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables from .env file
load_dotenv('../.env')

# Load OpenAI API key
api_key = os.getenv("OPENAI_API_KEY")
if api_key is None:
    logging.error("OpenAI API key is not set in the environment variables.")
    raise ValueError("OpenAI API key is not set in the environment variables.")
openai.api_key = api_key

# Load vector store from disk
try:
    with open('../data/vector_stored.pkl', 'rb') as f:
        vector_store = pickle.load(f)
    logging.info("Vector store loaded successfully.")
except FileNotFoundError:
    logging.error("Vector store file not found.")
    raise

# Load your SentenceTransformer model
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
logging.info("SentenceTransformer model loaded.")

# Perform PCA on embeddings
embeddings_np = np.array(vector_store['embeddings'])
pca = PCA(n_components=100)  # Adjust the number of components as needed
embeddings_reduced = pca.fit_transform(embeddings_np)
vector_store['embeddings_reduced'] = embeddings_reduced

# Setup FAISS index
d = embeddings_reduced.shape[1]  # Dimensionality of the reduced embeddings
index = faiss.IndexFlatL2(d)  # L2 distance for exact search
index.add(embeddings_reduced)  # Add the reduced embeddings to the index
logging.info("FAISS index created and embeddings added.")

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
        logging.debug(f"Added to history: Query: {query}, Answer: {answer}")

    def generate_answer_gpt4(self, relevant_documents, question):
        context = "\n\n".join([f"User: {entry['query']}\nAssistant: {entry['answer']}" for entry in self.conversation_history])
        context += "\n\n" + "\n\n".join(relevant_documents)
        prompt = f"Based on the following documents and conversation history, answer the question:\n\n{context}\n\nQuestion: {question}\nAnswer:"

        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.7  # Adjusted temperature for more creative responses
            )
            answer = response.choices[0].message['content'].strip()
            logging.info(f"Generated answer using GPT-4: {answer}")
            return answer
        except Exception as e:
            logging.error(f"Error generating answer: {e}")
            return "I'm sorry, I couldn't generate an answer. Please try again."

    def search_vector_store(self, query):
        try:
            query_embedding = self.model.encode(query)
            query_embedding_reduced = self.pca.transform([query_embedding]).astype('float32')
            D, I = self.index.search(query_embedding_reduced, self.top_k)
            results = [(self.vector_store['documents'][i], self.vector_store['metadatas'][i], D[0][n]) for n, i in enumerate(I[0])]
            logging.info(f"Found {len(results)} relevant documents for query: {query}")
            return results
        except Exception as e:
            logging.error(f"Error during vector store search: {e}")
            return []

    def process_query(self, query):
        if query.lower().strip() == "quit":
            logging.info("User initiated exit.")
            print("Ending conversation. Goodbye!")
            return None
        
        results = self.search_vector_store(query)
        if not results:
            return "I'm sorry, I couldn't find any relevant information. Please try rephrasing your question."

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
