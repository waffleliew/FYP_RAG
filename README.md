# Langchain RAG Pipeline

A single LLM RAG (Retrieval-Augmented Generation) pipeline built with Langchain, Pinecone, and Streamlit. This application allows you to upload documents, store them in a vector database, and query them using natural language. It serves as a prototype for building a more robust RAG system later on.

## Features

- **Document Processing**: Upload text files, PDFs, and Markdown documents or input text directly
- **Vector Storage**: Store document embeddings in Pinecone for efficient retrieval
- **Natural Language Querying**: Ask questions about your documents in natural language
- **Interactive UI**: User-friendly Streamlit interface

## Architecture

The application consists of two main components:

1. **RAG Agent (`rag_agent.py`)**: Handles document processing, vector storage, and query processing
   - Uses Groq's LLama-3.3-70b-versatile model for text generation
   - Integrates with Pinecone for vector storage and retrieval
   - Implements document loading and splitting

2. **Streamlit UI (`app.py`)**: Provides a web interface for interacting with the RAG agent
   - Configuration panel for setting up the RAG agent
   - Document upload and text input functionality
   - Chat interface for querying the system

## Prerequisites

- Python 3.12+
- Pinecone API key [(get it here)](https://docs.pinecone.io/guides/projects/manage-api-keys)
- Groq API key [(get it here)](https://console.groq.com/keys)

## Installation

1. Clone the repository

2. (Optional) Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
3. Install the required packages:
   ```bash
   pip install -r requirements.txt
4. Set up your Pinecone and Groq API keys in the `.env`
5. Run the Streamlit app:
   ```bash
   streamlit run app.py      
6. Specify a Unique Index Name in the UI to create a new index on Pinecone. (Note: If you want to use an existing index, specify here, and the app will load it from Pinecone instead of creating a new one.)
7. Upload a document or input text in the UI. (Note: Uploading a document or text will overwrite any existing data in the index.) 
8. Start Querying about your Documents!.
