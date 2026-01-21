from langchain_mistralai import MistralAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import SpacyTextSplitter
from src.read_input_data import read_and_process_inputdata
import pytest

@pytest.fixture(scope="session")
def processed_docs():
    file_path = "data/evenements-publics-openagenda.json"
    processed_docs = read_and_process_inputdata(file_path, debug=False)
    return processed_docs

@pytest.fixture(scope="session")
def chunked_docs(processed_docs):
    nlp_splitter = SpacyTextSplitter(chunk_size=1000, pipeline="fr_core_news_sm")
    chunks = nlp_splitter.split_documents(processed_docs)
    return chunks

@pytest.fixture(scope="session")
def vector_store():
    # Charge l'index FAISS pour l'ensemble des tests
    index_path = "vectorstore/faiss_openagenda_index"
    embeddings = MistralAIEmbeddings(model="mistral-embed")
    vector_store = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
    return vector_store