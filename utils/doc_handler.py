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
from langchain_core.embeddings import Embeddings
import torch
import os
import re
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SentenceTransformerWrapper(Embeddings):
    """Wrapper for SentenceTransformer to make it compatible with LangChain"""
    
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
    
    def embed_documents(self, texts):
        """Embed a list of texts."""
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()
    
    def embed_query(self, text):
        """Embed a single piece of text."""
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.tolist()

def get_embeddings(model_provider, base_url=None):
    """Helper function to get appropriate embeddings based on provider"""
    logger.info(f"Getting embeddings for provider: {model_provider}")
    try:
        if model_provider == "ollama" and base_url:
            logger.info("Attempting to use Ollama embeddings")
            embeddings = OllamaEmbeddings(
                model="nomic-embed-text",
                base_url=base_url
            )
            # Test Ollama embeddings
            test_text = "Test embedding generation"
            logger.info("Testing Ollama embeddings...")
            test_embedding = embeddings.embed_query(test_text)
            if not test_embedding:
                raise Exception("Ollama embedding generation failed")
            logger.info("Successfully created Ollama embeddings")
            return embeddings
    except Exception as e:
        logger.warning(f"Ollama embeddings failed: {str(e)}")
        st.warning(f"Falling back to all-MiniLM embeddings: {str(e)}")
    
    # Default/fallback to all-MiniLM
    logger.info("Using all-MiniLM embeddings")
    return SentenceTransformerWrapper()

def process_documents(uploaded_files, reranker, embedding_model, base_url):
    logger.info("Starting document processing")
    
    if st.session_state.documents_loaded:
        logger.info("Documents already loaded, skipping processing")
        return

    st.session_state.processing = True
    documents = []
    
    # Create temp directory
    if not os.path.exists("temp"):
        os.makedirs("temp")
        logger.info("Created temp directory")
    
    # Process files
    for file in uploaded_files:
        try:
            logger.info(f"Processing file: {file.name}")
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
                logger.warning(f"Unsupported file type: {file.name}")
                continue
                
            logger.info(f"Loading documents from {file.name}")
            documents.extend(loader.load())
            os.remove(file_path)
            logger.info(f"Successfully processed {file.name}")
        except Exception as e:
            logger.error(f"Error processing {file.name}: {str(e)}")
            st.error(f"Error processing {file.name}: {str(e)}")
            return

    # Text splitting
    logger.info("Splitting text into chunks")
    text_splitter = CharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separator="\n"
    )
    texts = text_splitter.split_documents(documents)
    text_contents = [doc.page_content for doc in texts]
    logger.info(f"Created {len(texts)} text chunks")

    try:
        # Get appropriate embeddings
        embeddings = get_embeddings("ollama" if base_url else "other", base_url)
        
        # Create vector store
        logger.info("Creating vector store")
        vector_store = FAISS.from_texts(
            texts=text_contents,
            embedding=embeddings,
            metadatas=[{"content": text} for text in text_contents]
        )
        logger.info("Successfully created vector store")
        
        # Create BM25 store
        logger.info("Creating BM25 retriever")
        bm25_retriever = BM25Retriever.from_texts(
            text_contents,
            bm25_impl=BM25Okapi,
            preprocess_func=lambda text: re.sub(r"\W+", " ", text).lower().split()
        )

        # Create ensemble retriever
        logger.info("Creating ensemble retriever")
        ensemble_retriever = EnsembleRetriever(
            retrievers=[
                bm25_retriever,
                vector_store.as_retriever(search_kwargs={"k": 5})
            ],
            weights=[0.4, 0.6]
        )

        # Build knowledge graph
        logger.info("Building knowledge graph")
        knowledge_graph = build_knowledge_graph(texts)

        # Store in session
        st.session_state.retrieval_pipeline = {
            "ensemble": ensemble_retriever,
            "reranker": reranker,
            "texts": text_contents,
            "knowledge_graph": knowledge_graph
        }

        st.session_state.documents_loaded = True
        
        # Debug information
        G = knowledge_graph
        logger.info(f"Knowledge graph stats - Nodes: {len(G.nodes)}, Edges: {len(G.edges)}")
        st.write(f"🔗 Total Nodes: {len(G.nodes)}")
        st.write(f"🔗 Total Edges: {len(G.edges)}")
        st.write(f"🔗 Sample Nodes: {list(G.nodes)[:5]}")
        st.write(f"🔗 Sample Edges: {list(G.edges)[:5]}")

    except Exception as e:
        logger.error(f"Error during document processing: {str(e)}", exc_info=True)
        st.error(f"Error during document processing: {str(e)}")
        st.session_state.processing = False
        return

    logger.info("Document processing completed successfully")
    st.session_state.processing = False