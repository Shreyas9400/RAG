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
import os
import re

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
            model = SentenceTransformer("all-MiniLM-L6-v2").to(device)
            embeddings = model

        # Vector store - process in smaller batches
        batch_size = 10
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            if isinstance(embeddings, SentenceTransformer):
                batch_embeddings = embeddings.encode([doc.page_content for doc in batch]).tolist()
                vector_store = FAISS.from_embeddings(
                    embeddings=batch_embeddings,
                    texts=[doc.page_content for doc in batch],
                    embedding=embeddings
                )
            else:
                vector_store = FAISS.from_documents(batch, embeddings)
            
            if i == 0:
                complete_vector_store = vector_store
            else:
                complete_vector_store.merge_from(vector_store)

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
                complete_vector_store.as_retriever(search_kwargs={"k": 5})
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