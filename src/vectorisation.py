import os
import numpy as np
from langchain_text_splitters import SpacyTextSplitter
from langchain_community.vectorstores import FAISS
from src.read_input_data import read_and_process_inputdata


def text_split_and_vectorize(documents, vector_store, index_path):
    nlp_splitter = SpacyTextSplitter(chunk_size=4000, pipeline="fr_core_news_sm")

    chunks = nlp_splitter.split_documents(documents)
    print(f"Number of chunks created: {len(chunks)}")
    
    # Vectorize and Create FAISS index
    if not vector_store.index.is_trained:
        print("Training FAISS index")
        training_text = [chunk.page_content for chunk in chunks]
        training_embedding = vector_store.embedding_function.embed_documents(training_text)
        training_vectors = np.array(training_embedding).astype("float32")
        vector_store.index.train(training_vectors)
        
    vector_store.add_documents(chunks)
    vector_store.save_local(index_path)
    print(f"Vector database saved to {index_path}")
    return vector_store

def get_vector_store(embedding_dim, index_path):
    if os.path.exists(index_path):
        vector_store = FAISS.load_local(index_path, embedding_dim, allow_dangerous_deserialization=True)
        print(f"Loaded existing vector database from {index_path}")
    else:
        print(f"Creating a new vector db:")  
        # Read and process data
        processed_docs = read_and_process_inputdata('data/evenements-publics-openagenda-full.json', debug=True)
        print(f"\nNumber of documents: {len(processed_docs)}")
        print(f"\nFirst Doc: {processed_docs[0]}")
        
        # Split documents into chunks and create vector database
        vector_store = text_split_and_vectorize(processed_docs, vector_store, index_path)
        print(f"Vector database saved and ready to use.")
    return vector_store

