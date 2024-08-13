from vector_utils import load_vector_store, find_similar
from openai_utils import generate_answer_gpt4

def main():
    # Load vector store
    vector_store_path = './data/vector_store.json'
    vector_store = load_vector_store(vector_store_path)
    
    print("Welcome to the RGU Campus Query Bot! Type 'exit' to end the chat.")
    
    while True:
        query = input("You: ")
        if query.lower() == 'exit':
            print("Goodbye and Feel welcomed at RGU!")
            break
        
        # Find similar documents
        similar_docs = find_similar(query, vector_store)
        
        if not similar_docs:
            print("I'm sorry, I couldn't find relevant information.")
            continue
        
        # Generate and return the answer
        answer = generate_answer_gpt4(similar_docs, query)
        print(f"rgu_bot: {answer}")

if __name__ == "__main__":
    main()

