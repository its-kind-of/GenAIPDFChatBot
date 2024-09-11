from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter

class PDFProcessor:
    def __init__(self, chunk_size=1000, chunk_overlap=200):
        # Configurable chunk size and overlap for text splitting
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def extract_text(self, pdf_docs):
        """Extract text from multiple PDF files."""
        text = ""
        for pdf in pdf_docs:
            reader = PdfReader(pdf)
            for page in reader.pages:
                text += page.extract_text()
        return text

    def split_text(self, text):
        """Split the extracted text into smaller chunks."""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        return splitter.split_text(text)
