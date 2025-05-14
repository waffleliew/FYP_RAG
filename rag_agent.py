import os
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.embeddings import Embeddings
from langchain.embeddings import HuggingFaceEmbeddings

from pinecone_store import PineconeStore
from document_loader import DocumentLoader

# Load environment variables
load_dotenv()

class RAGAgent:
    def __init__(
        self,
        index_name: str,
        namespace: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ):
        """Initialize the RAG agent.
        
        Args:
            index_name: Name of the Pinecone index
            namespace: Optional namespace for the vectors
            embeddings: Optional embedding model
            system_prompt: Optional system prompt for the LLM
        """
        # Initialize the LLM
        self.llm = ChatGroq(model="llama-3.3-70b-versatile") 
        
        # Initialize Pinecone store
        self.vector_store = PineconeStore(
            index_name=index_name,
            namespace=namespace)
        
        # Initialize document loader
        self.document_loader = DocumentLoader()
        
        # Set up the default system prompt if not provided
        if system_prompt is None:
            self.system_prompt = """You are a helpful assistant that answers questions based on the provided context. 
            If you don't know the answer or can't find it in the context, say so instead of making up information.
            Always cite your sources when possible."""
        else:
            self.system_prompt = system_prompt
        
        # Set up the RAG chain
        self._setup_rag_chain()
    
    def _setup_rag_chain(self):
        """Set up the RAG chain."""
        # Define the prompt template
        prompt_template = ChatPromptTemplate([
            ("system", self.system_prompt),
            ("system", "Context information is below:\n\n{context}"),
            ("human", "{question}")
        ])
        
        # Define the retrieval function
        def retrieve_context(query: str) -> List[str]:
            docs = self.vector_store.similarity_search(query)
            print(f"Found {len(docs)} documents")

            return [doc.page_content for doc in docs]
        # Set up the RAG chain
        self.rag_chain = (
            {"context": retrieve_context, "question": RunnablePassthrough()}
            | prompt_template
            | self.llm
            | StrOutputParser()
        )


    def add_documents(self, documents: List[Any]) -> None:
        """Add documents to the vector store.
        
        Args:
            documents: List of documents to add
        """
        self.vector_store.add_documents(documents)
    
    def load_and_add_text(self, text: str, metadata: Optional[dict] = None) -> None:
        """Load, split, and add text to the vector store.
        
        Args:
            text: Text content
            metadata: Optional metadata for the document
        """
        documents = self.document_loader.load_and_split_text(text, metadata)
        self.add_documents(documents)
    
    def load_and_add_file(self, file_path: str) -> None:
        """Load, split, and add a text file to the vector store.
        
        Args:
            file_path: Path to the text file
        """
        documents = self.document_loader.load_and_split_file(file_path)
        self.add_documents(documents)
    
    def load_and_add_directory(self, directory_path: str, extensions: List[str] = ['.txt', '.md']) -> None:
        """Load, split, and add all text files in a directory to the vector store.
        
        Args:
            directory_path: Path to the directory
            extensions: List of file extensions to include
        """
        documents = self.document_loader.load_and_split_directory(directory_path, extensions)
        self.add_documents(documents)
    
    def query(self, question: str) -> str:
        """Query the RAG system.
        
        Args:
            question: Question to ask
            
        Returns:
            Answer from the RAG system
        """
        return self.rag_chain.invoke(question)