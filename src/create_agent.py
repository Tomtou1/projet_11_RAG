import os
import faiss
import streamlit as st
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain.tools import tool
from langchain.agents import create_agent

from src.read_input_data import read_and_process_inputdata
from src.vectorisation import text_split_and_vectorize

def initialize_rag_system(has_streamlit=True):

    # Load keys, model, embeddings
    load_dotenv()
    index_path = "vectorstore/faiss_openagenda_index"
    model = init_chat_model("mistral-small-latest", model_provider="mistralai")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    embedding_dim = len(embeddings.embed_query("hello world"))
    index = faiss.IndexFlatL2(embedding_dim)

    vector_store = FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
    )
    
    # Load existing vector database or create a new one
    if os.path.exists(index_path):
        vector_store = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
        if has_streamlit:
            st.success(f"Loaded existing vector database")
    else:
        if has_streamlit:
            st.warning(f"Creating a new vector database...")
        # Read / Process / Split and Vectorize data
        processed_docs = read_and_process_inputdata('data/evenements-publics-openagenda-full.json', debug=False)
        vector_store = text_split_and_vectorize(processed_docs, vector_store, index_path)
        if has_streamlit:
            st.success(f"Vector database created and ready to use")

    @tool(response_format="content_and_artifact")
    def retrieve_context(query: str):
        """Retrieve information to help answer a query."""
        retrieved_docs = vector_store.similarity_search(query, k=5)
        serialized = "\n\n".join(
            (f"Source: {doc.metadata}\nContent: {doc.page_content}")
            for doc in retrieved_docs
        )
        return serialized, retrieved_docs
    
    tools = [retrieve_context]
    system_prompt = (
        "You are giving recommendations for cultural events. "
        "Answer in French and be helpful and enthusiastic about cultural activities. "
        "Use the retrieved context to provide specific and accurate information about events."
        "Be short and precise in your answer, do not give answers that are not from the retrieved context"
        "The context is about 2025 events, so you need to act like we are right now beginning of the year 2025"
    )
    agent = create_agent(model, tools, system_prompt=system_prompt)
    return agent, vector_store