import os
import faiss
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_mistralai import MistralAIEmbeddings
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain.tools import tool
from langchain.agents import create_agent

from src.read_input_data import read_and_process_inputdata
from src.vectorisation import text_split_and_vectorize

if __name__ == "__main__":
    
    # Load keys, model, embeddings, and create vector store
    load_dotenv()
    index_path = "vectorstore/faiss_openagenda_index"
    model = init_chat_model("mistral-large-latest", model_provider="mistralai")
    embeddings = MistralAIEmbeddings(model="mistral-embed")

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
        print(f"Loaded existing vector database from {index_path}")
    else:
        print(f"Creating a new vector db:")  
        # Read / Process / Split and Vectorize data
        processed_docs = read_and_process_inputdata('data/evenements-publics-openagenda.json', debug=True)
        print(f"\nNumber of documents: {len(processed_docs)}")
        print(f"\nFirst Doc: {processed_docs[0]}")
        vector_store = text_split_and_vectorize(processed_docs, vector_store, index_path)
        print(f"Vector database saved and ready to use.")