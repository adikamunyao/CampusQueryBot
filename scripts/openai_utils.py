import openai
import os

# Load OpenAI API key
api_key = os.getenv("OPENAI_API_KEY")
openai.api_key = api_key

def generate_answer_gpt4(relevant_documents, question):
    """Generate an answer using OpenAI's GPT-4 model."""
    context = "\n\n".join(relevant_documents)
    prompt = f"Based on the following documents, answer the question:\n\n{context}\n\nQuestion: {question}\nAnswer:"
    
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini", 
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=150,
        temperature=0.7
    )
    
    return response.choices[0].message['content'].strip()

