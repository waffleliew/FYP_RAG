import os
import streamlit as st
from dotenv import load_dotenv
import tempfile
from rag_agent import RAGAgent

# Load environment variables
load_dotenv()

# Set page configuration
st.set_page_config(
    page_title="RAG Agent with Pinecone",
    layout="wide"
)

# Initialize session state variables
if "messages" not in st.session_state:
    st.session_state.messages = []
if "rag_agent" not in st.session_state:
    st.session_state.rag_agent = None

# Sidebar for configuration and document upload
with st.sidebar:
    st.title("RAG Agent Configuration")
    # Pinecone configuration
    st.subheader("Pinecone Settings")
    index_name = st.text_input("Index Name", "langchain-rag-demo")
    namespace = st.text_input("Namespace (Optional)", "streamlit-ui")
    
    # Initialize RAG agent button initializes the following instances: 1. LLM, 2. Embedding, 3. PinconeStore, 4. DocumentLoader 5. RAG Chain
    if st.button("Initialize RAG Agent"):
        with st.spinner("Initializing RAG Agent..."):
            try:
                st.session_state.rag_agent = RAGAgent(
                    index_name=index_name,
                    namespace=namespace
                )
                st.success("RAG Agent initialized successfully!")
            except Exception as e:
                st.error(f"Error initializing RAG Agent: {str(e)}")
    

    # Document upload section
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    st.subheader("Upload Documents")
    # File upload
    uploaded_files = st.file_uploader(
        "Choose files", 
        accept_multiple_files=True,
        type=["txt", "md", "pdf"]
    )
    
    if uploaded_files:
        if st.button("Process Files") and st.session_state.rag_agent is not None:
            with st.spinner("Processing files..."):
                for uploaded_file in uploaded_files:
                    # Create a temporary file
                    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_path = tmp_file.name
                    
                    try:
                        st.session_state.rag_agent.load_and_add_file(tmp_path)
                        st.success(f"File '{uploaded_file.name}' processed successfully!")

                    except Exception as e:
                        st.error(f"Error processing file '{uploaded_file.name}': {str(e)}")

                    finally:
                        # Clean up the temporary file
                        os.unlink(tmp_path)
   
    st.write("Or")

    # Text input
    text_input = st.text_area("Enter text content", height=100)
    text_source = st.text_input("Source name (metadata)")
    
    if st.button("Add Text") and st.session_state.rag_agent is not None:
        if text_input.strip():
            with st.spinner("Processing text..."):
                try:
                    st.session_state.rag_agent.load_and_add_text(
                        text_input, 
                        metadata={"source": text_source if text_source else "direct input"}
                    )
                    st.success("Text added successfully!")
                except Exception as e:
                    st.error(f"Error adding text: {str(e)}")
        else:
            st.warning("Please enter some text.")
    

# Main chat interface
st.title("Chat with RAG Agent")

#Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Chat input
if prompt := st.chat_input("Ask a question..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.write(prompt)
    
    # Generate and display assistant response
    if st.session_state.rag_agent is not None:
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = st.session_state.rag_agent.query(prompt)
                    st.write(response)

                    # Add assistant response to chat history
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    error_msg = f"Error generating response: {str(e)}"
                    st.error(error_msg)
                    # Add error message to chat history
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
    else:
        with st.chat_message("assistant"):
            error_msg = "Please initialize the RAG Agent in the sidebar first."
            st.warning(error_msg)
            # Add warning message to chat history
            st.session_state.messages.append({"role": "assistant", "content": error_msg})