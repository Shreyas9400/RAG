import streamlit as st
import json
from utils.retriever_pipeline import retrieve_documents
from utils.doc_handler import process_documents
from utils.model_utils import ModelProvider
from sentence_transformers import CrossEncoder
import torch
import os

# Define device before using it
device = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize Cross-Encoder (Reranker)
reranker = None
try:
    # With this:
    reranker = CrossEncoder('sentence-transformers/all-MiniLM-L6-v2', device=device)
except Exception as e:
    st.error(f"Failed to load CrossEncoder model: {str(e)}")

# Add provider configuration to sidebar
with st.sidebar:
    st.header("🤖 Model Provider")
    model_provider = st.selectbox(
        "Select Model Provider", 
        ["Ollama", "Gemini", "Groq", "OpenAI"]
    )

    # Model selection based on provider
    if model_provider == "Ollama":
        MODEL = st.selectbox("Ollama Model", ["deepseek-r1:7b", "llama2", "mistral"])
        EMBEDDINGS_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
        OLLAMA_BASE_URL = "http://localhost:11434"  # Changed from base_url to OLLAMA_BASE_URL
        api_key = None
    elif model_provider == "Gemini":
        MODEL = st.selectbox("Gemini Model", ["gemini-2.0-flash"])
        EMBEDDINGS_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
        api_key = st.text_input("Google API Key", type="password")
        OLLAMA_BASE_URL = None
    elif model_provider == "Groq":
        MODEL = st.selectbox("Groq Model", ["llama2-70b-4096", "mixtral-8x7b-32768"])
        EMBEDDINGS_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
        api_key = st.text_input("Groq API Key", type="password")
        OLLAMA_BASE_URL = None
    elif model_provider == "OpenAI":
        MODEL = st.selectbox("OpenAI Model", ["gpt-3.5-turbo", "gpt-4"])
        EMBEDDINGS_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
        api_key = st.text_input("OpenAI API Key", type="password")
        OLLAMA_BASE_URL = None


# Custom CSS
st.markdown("""
    <style>
        .stApp { background-color: #f4f4f9; }
        h1 { color: #00FF99; text-align: center; }
        .stChatMessage { border-radius: 10px; padding: 10px; margin: 10px 0; }
        .stChatMessage.user { background-color: #e8f0fe; }
        .stChatMessage.assistant { background-color: #d1e7dd; }
        .stButton>button { background-color: #00AAFF; color: white; }
    </style>
""", unsafe_allow_html=True)

                                                                                    # Manage Session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "retrieval_pipeline" not in st.session_state:
    st.session_state.retrieval_pipeline = None
if "rag_enabled" not in st.session_state:
    st.session_state.rag_enabled = False
if "documents_loaded" not in st.session_state:
    st.session_state.documents_loaded = False


with st.sidebar:                                                                        # 📁 Sidebar
    st.header("📁 Document Management")
    uploaded_files = st.file_uploader(
        "Upload documents (PDF/DOCX/TXT)",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True
    )
    
    if uploaded_files and not st.session_state.documents_loaded:
        with st.spinner("Processing documents..."):
            process_documents(uploaded_files,reranker,EMBEDDINGS_MODEL, OLLAMA_BASE_URL)
            st.success("Documents processed!")
    
    st.markdown("---")
    st.header("⚙️ RAG Settings")
    
    st.session_state.rag_enabled = st.checkbox("Enable RAG", value=True)
    st.session_state.enable_hyde = st.checkbox("Enable HyDE", value=True)
    st.session_state.enable_reranking = st.checkbox("Enable Neural Reranking", value=True)
    st.session_state.enable_graph_rag = st.checkbox("Enable GraphRAG", value=True)
    st.session_state.temperature = st.slider("Temperature", 0.0, 1.0, 0.3, 0.05)
    st.session_state.max_contexts = st.slider("Max Contexts", 1, 5, 3)
    
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

    # 🚀 Footer (Bottom Right in Sidebar) For some Credits :)
    st.sidebar.markdown("""
        <div style="position: absolute; top: 20px; right: 10px; font-size: 12px; color: gray;">
            <b>Developed by:</b> N Sai Akhil &copy; All Rights Reserved 2025
        </div>
    """, unsafe_allow_html=True)

# 💬 Chat Interface
st.title("🤖 DeepGraph RAG-Pro")
st.caption("Advanced RAG System with GraphRAG, Hybrid Retrieval, Neural Reranking and Chat History")

# Display messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# In the main chat generation section, replace the Ollama-specific code with:
# Replace OLLAMA_API_URL references with this check:
if prompt := st.chat_input("Ask about your documents..."):
    # Instantiate ModelProvider
    model_provider_instance = ModelProvider(
        provider=model_provider.lower(), 
        model=MODEL, 
        api_key=api_key
    )    

    # Generate response
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""
        
        # Build context
        context = ""
        if st.session_state.rag_enabled and st.session_state.retrieval_pipeline:
            try:
                # For non-Ollama providers, use a different URL or method
                if model_provider.lower() == "ollama":
                    docs = retrieve_documents(prompt, f"{OLLAMA_BASE_URL}/api/generate", MODEL, chat_history)
                else:
                    docs = retrieve_documents(prompt, None, MODEL, chat_history)
                context = "\n".join(
                    f"[Source {i+1}]: {doc.page_content}" 
                    for i, doc in enumerate(docs)
                )
            except Exception as e:
                st.error(f"Retrieval error: {str(e)}")
        
        # 🚀 Structured Prompt
        system_prompt = f"""Use the chat history to maintain context:
            Chat History:
            {chat_history}

            Analyze the question and context through these steps:
            1. Identify key entities and relationships
            2. Check for contradictions between sources
            3. Synthesize information from multiple contexts
            4. Formulate a structured response

            Context:
            {context}

            Question: {prompt}
            Answer:"""
        
        # Stream response
        response = requests.post(
            OLLAMA_API_URL,
            json={
                "model": MODEL,
                "prompt": system_prompt,
                "stream": True,
                "options": {
                    "temperature": st.session_state.temperature,  # Use dynamic user-selected value
                    "num_ctx": 4096
                }
            },
            stream=True
        )
        try:
            for line in response.iter_lines():
                if line:
                    data = json.loads(line.decode())
                    token = data.get("response", "")
                    full_response += token
                    response_placeholder.markdown(full_response + "▌")
                    
                    # Stop if we detect the end token
                    if data.get("done", False):
                        break
                        
            response_placeholder.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            
        except Exception as e:
            st.error(f"Generation error: {str(e)}")
            st.session_state.messages.append({"role": "assistant", "content": "Sorry, I encountered an error."})
