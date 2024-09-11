import pinecone
import requests
from langchain.embeddings.base import Embeddings
from langchain.vectorstores import Pinecone
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
import os

class OllamaEmbedding(Embeddings):
    def __init__(self):
        self.server_url = os.getenv("OLLAMA_SERVER_URL")
    
    def embed_documents(self, texts):
        """Embed documents using the local Ollama server."""
        embeddings = []
        for text in texts:
            response = requests.post(
                self.server_url,
                json={
                    'model': 'ollama3:latest',
                    'prompt': text,
                    'max_tokens': 0  # No text generation required
                }
            )
            if response.status_code != 200:
                raise Exception(f"API request failed: {response.status_code}")
            embeddings.append(response.json()['embedding'])
        return embeddings

class Chatbot:
    def __init__(self):
        load_dotenv()
        self.index_name = "pdf-embeddings-index"
        self.embedding_model = OllamaEmbedding()
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

        # Initialize Pinecone
        pinecone.init(api_key=os.getenv("PINECONE_API_KEY"), environment=os.getenv("PINECONE_ENV"))

        # Create or connect to the Pinecone index
        if self.index_name not in pinecone.list_indexes():
            pinecone.create_index(self.index_name, dimension=1536)
        self.index = pinecone.Index(self.index_name)
    
    def store_embeddings(self, text_chunks):
        """Store text chunks in Pinecone as embeddings."""
        embeddings = self.embedding_model.embed_documents(text_chunks)
        # Upsert embeddings into Pinecone
        for i, embedding in enumerate(embeddings):
            self.index.upsert([(str(i), embedding)])

    def query_pinecone(self, user_question):
        """Query Pinecone to find relevant document contexts."""
        # Embed the user question
        question_embedding = self.embedding_model.embed_documents([user_question])[0]
        # Query Pinecone for the closest matches
        result = self.index.query(queries=[question_embedding], top_k=5, include_metadata=True)
        return [match['metadata']['text'] for match in result['matches']]

    def add_to_memory(self, question, answer):
        """Add the conversation to memory."""
        self.memory.save_context({"question": question}, {"answer": answer})

    def get_memory_context(self):
        """Retrieve the conversation history."""
        return "\n".join([f"Q: {turn['question']}\nA: {turn['answer']}" for turn in self.memory.buffer])

    def generate_response(self, user_question):
        """Generate response using Ollama Serve and Pinecone results."""
        # Query Pinecone to get relevant document chunks
        document_contexts = self.query_pinecone(user_question)
        memory_context = self.get_memory_context()

        # Build the prompt for Ollama Serve
        prompt = (
            f"Context: {document_contexts}\n\n"
            f"Conversation history: {memory_context}\n\n"
            f"User question: {user_question}\n\nAnswer:"
        )

        # Request response from Ollama Serve
        response = requests.post(
            os.getenv("OLLAMA_SERVER_URL"),
            json={
                'model': 'ollama3:latest',
                'prompt': prompt,
                'max_tokens': 150,
                'temperature': 0.7
            }
        )
        if response.status_code != 200:
            raise Exception(f"API request failed: {response.status_code}")
        
        answer = response.json()['completion']
        
        # Update the conversation memory
        self.add_to_memory(user_question, answer)

        return answer
