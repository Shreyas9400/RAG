# In utils/doc_handler.py - Update the vector store creation logic:

import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from utils.build_graph import build_knowledge_graph
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import torch
import os
import re
import numpy as np

def process_documents(uploaded_files, reranker, embedding_model, base_url):
    if st.session_state.documents_loaded:
        return

    st.session_state.processing = True
    documents = []
    
    # Create temp directory
    if not os.path.exists("temp"):
        os.makedirs("temp")
    
    # Process files
    for file in uploaded_files:
        try:
            file_path = os.path.join("temp", file.name)
            with open(file_path, "wb") as f:
                f.write(file.getbuffer())

            if file.name.endswith(".pdf"):
                loader = PyPDFLoader(file_path)
            elif file.name.endswith(".docx"):
                loader = Docx2txtLoader(file_path)
            elif file.name.endswith(".txt"):
                loader = TextLoader(file_path)
            else:
                continue
                
            documents.extend(loader.load())
            os.remove(file_path)
        except Exception as e:
            st.error(f"Error processing {file.name}: {str(e)}")
            return

    # Text splitting
    text_splitter = CharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separator="\n"
    )
    texts = text_splitter.split_documents(documents)
    text_contents = [doc.page_content for doc in texts]

    try:
        # Initialize embeddings based on availability
        try:
            embeddings = OllamaEmbeddings(
                model=embedding_model,
                base_url=base_url
            )
            # Test Ollama embeddings
            test_embedding = embeddings.embed_documents([text_contents[0]])
            if not test_embedding:
                raise Exception("Ollama embedding generation failed")
        except Exception as e:
            st.warning(f"Falling back to all-MiniLM embeddings: {str(e)}")
            device = "cuda" if torch.cuda.is_available() else "cpu"
            embeddings_model = SentenceTransformer("all-MiniLM-L6-v2").to(device)
            
            # Create embeddings for all texts
            text_embeddings = embeddings_model.encode(text_contents)
            
            # Create metadatas for all texts
            metadatas = [{"content": text} for text in text_contents]
            
            # Initialize FAISS index
            vector_store = FAISS.from_embeddings(
                text_embeddings=text_embeddings.tolist(),  # Convert numpy array to list
                texts=text_contents,
                metadatas=metadatas,
                embedding=embeddings_model
            )
        else:
            # If Ollama embeddings worked, use them directly
            vector_store = FAISS.from_documents(texts, embeddings)

        # BM25 store
        bm25_retriever = BM25Retriever.from_texts(
            text_contents,
            bm25_impl=BM25Okapi,
            preprocess_func=lambda text: re.sub(r"\W+", " ", text).lower().split()
        )

        # Ensemble retrieval
        ensemble_retriever = EnsembleRetriever(
            retrievers=[
                bm25_retriever,
                vector_store.as_retriever(search_kwargs={"k": 5})
            ],
            weights=[0.4, 0.6]
        )

        # Store in session
        st.session_state.retrieval_pipeline = {
            "ensemble": ensemble_retriever,
            "reranker": reranker,
            "texts": text_contents,
            "knowledge_graph": build_knowledge_graph(texts)
        }

        st.session_state.documents_loaded = True
        
        # Debug information
        G = st.session_state.retrieval_pipeline["knowledge_graph"]
        st.write(f"🔗 Total Nodes: {len(G.nodes)}")
        st.write(f"🔗 Total Edges: {len(G.edges)}")
        st.write(f"🔗 Sample Nodes: {list(G.nodes)[:10]}")
        st.write(f"🔗 Sample Edges: {list(G.edges)[:10]}")

    except Exception as e:
        st.error(f"Error during document processing: {str(e)}")
        st.session_state.processing = False
        return

    st.session_state.processing = False