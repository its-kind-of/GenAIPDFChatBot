import streamlit as st
from pdf_processor import PDFProcessor
from chatbot import Chatbot

class PDFChatApp:
    def __init__(self):
        self.pdf_processor = PDFProcessor()
        self.chatbot = Chatbot()

    def run(self):
        st.set_page_config(page_title="Chat with PDF using Ollama Serve and Pinecone", page_icon=":file_pdf:")

        # UI Layout
        st.title("Chat with your PDFs using Ollama Serve and Pinecone")

        # User input for the question
        user_question = st.text_input("Ask a question related to your PDFs:")

        if user_question:
            # Generate the response
            response = self.chatbot.generate_response(user_question)
            st.write("Response:", response)

        # Sidebar for PDF upload
        with st.sidebar:
            st.header("Upload your PDF(s)")
            pdf_docs = st.file_uploader("Upload PDF Files", accept_multiple_files=True)

            if st.button("Submit & Process"):
                with st.spinner("Processing PDFs..."):
                    # Extract and process text from PDFs
                    raw_text = self.pdf_processor.extract_text(pdf_docs)
                    text_chunks = self.pdf_processor.split_text(raw_text)

                    # Store embeddings in Pinecone
                    self.chatbot.store_embeddings(text_chunks)
                st.success("PDFs processed successfully!")

if __name__ == "__main__":
    # Start the PDF chat application
    app = PDFChatApp()
    app.run()
