import os
import openai
import faiss
import pickle
import logging
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from django.shortcuts import render
from django.http import JsonResponse

# Load environment variables from .env file
load_dotenv('../.env')

# Load OpenAI API key
api_key = os.getenv("OPENAI_API_KEY")
if api_key is None:
    logging.error("OpenAI API key is not set in the environment variables.")
    raise ValueError("OpenAI API key is not set in the environment variables.")
openai.api_key = api_key

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
            return None
        
        results = self.search_vector_store(query)
        if not results:
            return "I'm sorry, I couldn't find any relevant information. Please try rephrasing your question."

        relevant_documents = [doc for doc, _, _ in results]
        answer = self.generate_answer_gpt4(relevant_documents, query)
        
        # Add the query and answer to the conversation history
        self.add_to_history(query, answer)
        
        return answer

# Initialize and load necessary components
try:
    with open('../data/pca_model.pkl', 'rb') as f:
        pca = pickle.load(f)
    logging.info("PCA model loaded successfully.")
    
    index = faiss.read_index('../data/faiss_index.idx')
    logging.info("FAISS index loaded successfully.")
    
    with open('../data/vector_store_reduced.pkl', 'rb') as f:
        vector_store = pickle.load(f)
    logging.info("Reduced vector store loaded successfully.")
    
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    logging.info("SentenceTransformer model loaded.")
    
    qa_agent = QAChatAgent(model, pca, index, vector_store)

except Exception as e:
    logging.error(f"Error during initialization: {e}")
    raise RuntimeError("Failed to initialize the chatbot components.") from e

def chatbot_view(request):
    if request.method == 'POST':
        user_message = request.POST.get('message')
        
        if user_message:
            response = qa_agent.process_query(user_message)
            return JsonResponse({'response': response})
        else:
            return JsonResponse({'error': 'No message provided.'}, status=400)
    
    return render(request, 'chatbot/chat.html')  # Render your chatbot template

