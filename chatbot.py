import os
import time
from pinecone import Pinecone, ServerlessSpec
from langchain.embeddings.base import Embeddings
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
import requests

# Load environment variables
load_dotenv()

class OllamaEmbedding(Embeddings):
    """Custom embedding class using Ollama Serve for embeddings."""
    def __init__(self):
        self.server_url = os.getenv("OLLAMA_SERVER_URL")

    def embed_documents(self, texts):
        """Embed documents using Ollama Serve."""
        embeddings = []
        for text in texts:
            response = requests.post(
                self.server_url,
                json={'model': 'ollama3:latest', 'prompt': text, 'max_tokens': 0}
            )
            if response.status_code != 200:
                raise Exception(f"API request failed: {response.status_code}")
            embeddings.append(response.json()['embedding'])
        return embeddings

    def embed_query(self, query):
        """Embed a single query using Ollama Serve."""
        response = requests.post(
            self.server_url,
            json={'model': 'ollama3:latest', 'prompt': query, 'max_tokens': 0}
        )
        if response.status_code != 200:
            raise Exception(f"API request failed: {response.status_code}")
        return response.json()['embedding']

class Chatbot:
    """Chatbot class for embedding, querying Pinecone, and managing memory."""
    def __init__(self):
        # Initialize Pinecone with API key and region
        self.api_key = os.getenv("PINECONE_API_KEY")
        self.environment = "us-east-1"  # Adjust this if necessary
        self.pinecone = Pinecone(api_key=self.api_key)

        self.index_name = "pdf-embeddings-index"
        self.embedding_model = OllamaEmbedding()
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

        # Check if index exists, and delete it if necessary
        self.delete_index_if_exists()

        # Create a new index
        self.create_new_index()

        # Connect to the newly created index
        self.index = self.pinecone.Index(self.index_name)

    def delete_index_if_exists(self):
        """Delete the index if it already exists."""
        indexes = self.pinecone.list_indexes()  # Returns a list of index names
        if self.index_name in indexes:
            print(f"Index '{self.index_name}' already exists. Deleting it for recreation...")
            self.pinecone.delete_index(self.index_name)

            # Wait for 5 seconds to ensure the index is properly deleted
            time.sleep(5)
            print(f"Index '{self.index_name}' deleted successfully.")

    def create_new_index(self):
        """Create a new index with the given parameters."""
        print(f"Creating new index '{self.index_name}'...")
        self.pinecone.create_index(
            name=self.index_name,
            dimension=1536,  # Assuming this is the correct dimension for your embedding model
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",   # Ensure correct cloud provider
                region=self.environment  # Use correct region
            )
        )
        print(f"Index '{self.index_name}' created successfully.")

    def store_embeddings(self, text_chunks):
        """Store text chunk embeddings in Pinecone."""
        embeddings = self.embedding_model.embed_documents(text_chunks)
        for i, embedding in enumerate(embeddings):
            self.index.upsert([(str(i), embedding)])

    def query_pinecone(self, user_question):
        """Query Pinecone to find relevant document contexts."""
        question_embedding = self.embedding_model.embed_query(user_question)
        result = self.index.query(queries=[question_embedding], top_k=5, include_metadata=True)
        return [match['metadata']['text'] for match in result['matches']]

    def add_to_memory(self, question, answer):
        """Add the conversation to memory."""
        self.memory.save_context({"question": question}, {"answer": answer})

    def get_memory_context(self):
        """Retrieve the conversation history."""
        return "\n".join([f"Q: {turn['question']}\nA: {turn['answer']}" for turn in self.memory.buffer])

    def generate_response(self, user_question):
        """Generate a response using Ollama Serve and Pinecone results."""
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
            json={'model': 'ollama3:latest', 'prompt': prompt, 'max_tokens': 150, 'temperature': 0.7}
        )
        if response.status_code != 200:
            raise Exception(f"API request failed: {response.status_code}")
        
        answer = response.json()['completion']
        self.add_to_memory(user_question, answer)
        return answer
