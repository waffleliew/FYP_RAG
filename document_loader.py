from typing import List, Optional
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

class DocumentLoader:
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ):
        """Initialize the document loader.
        
        Args:
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
        """
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
    
    def load_and_split_text(self, text: str, metadata: Optional[dict] = None) -> List[Document]:
        """Load and split text into documents.
        
        Args:
            text: Text content
            metadata: Optional metadata for the document
            
        Returns:
            List of split documents
        """
        if metadata is None:
            metadata = {}
        
        doc = Document(page_content=text, metadata=metadata)
        return self.text_splitter.split_documents([doc])
    
    def load_and_split_file(self, file_path: str) -> List[Document]:
        """Load and split a file into documents.
        
        Args:
            file_path: Path to the file
            
        Returns:
            List of split documents
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Get file extension
        _, ext = os.path.splitext(file_path)
        ext = ext.lower()
        
        # Handle different file types
        if ext == '.pdf':
            try:
                # Import PDF loader only when needed
                from langchain_community.document_loaders import PyPDFLoader
                
                loader = PyPDFLoader(file_path)
                documents = loader.load()
                
                # Add source metadata
                for doc in documents:
                    doc.metadata["source"] = file_path
                
                # Split the documents
                return self.text_splitter.split_documents(documents)
            except ImportError:
                raise ImportError("PyPDF dependency not found. Install it with 'pip install pypdf'")
        else:
            # Handle text files
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
            except UnicodeDecodeError:
                # Try with a different encoding if UTF-8 fails
                with open(file_path, 'r', encoding='latin-1') as f:
                    text = f.read()
        
            metadata = {"source": file_path}
            return self.load_and_split_text(text, metadata)
    
    def load_and_split_directory(self, directory_path: str, extensions: List[str] = ['.txt', '.md', '.pdf']) -> List[Document]:
        """Load and split all files in a directory.
        
        Args:
            directory_path: Path to the directory
            extensions: List of file extensions to include
            
        Returns:
            List of split documents
        """
        if not os.path.exists(directory_path):
            raise FileNotFoundError(f"Directory not found: {directory_path}")
        
        documents = []
        for root, _, files in os.walk(directory_path):
            for file in files:
                if any(file.endswith(ext) for ext in extensions):
                    file_path = os.path.join(root, file)
                    try:
                        documents.extend(self.load_and_split_file(file_path))
                    except Exception as e:
                        print(f"Error processing file '{file}': {str(e)}")
        
        return documents