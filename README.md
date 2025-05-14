# Langchain RAG Pipeline

A single LLM RAG (Retrieval-Augmented Generation) pipeline built with Langchain, Pinecone, and Streamlit. This application allows you to upload documents, store them in a vector database, and query them using natural language.

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
- Pinecone API key
- Groq API key

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd Langchain-RAG

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