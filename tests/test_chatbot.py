from src.read_input_data import read_and_process_inputdata
from langchain_community.vectorstores import FAISS

    
# Numbers of vectors in Faiss index
def test_nbr_vectors(vector_store, file_path):
    #check nbr docs processed
    processed_docs = read_and_process_inputdata(file_path, debug=False)
    assert len(processed_docs) == vector_store.index.ntotal, f"Expected {len(processed_docs)} vectors, got {vector_store.index.ntotal}"
