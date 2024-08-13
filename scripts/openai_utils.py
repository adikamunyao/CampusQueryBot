import os
import openai
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv('./.env')

# Load OpenAI API key
api_key = os.getenv("OPENAI_API_KEY")
if api_key is None:
    raise ValueError("OpenAI API key is not set in the environment variables.")
openai.api_key = api_key
print('key loaded')
print('Model settimg up...')
def generate_answer_gpt4(relevant_documents, question):
    # Combine relevant documents into a single context
    context = "\n\n".join(relevant_documents)
    
    # Create a prompt for OpenAI
    prompt = f"Based on the following documents, answer the question:\n\n{context}\n\nQuestion: {question}\nAnswer:"
    
    # Generate the response from OpenAI using the chat endpoint
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini-2024-07-18", 
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=150,
        temperature=0.8
    )
    
    return response.choices[0].message['content'].strip()

