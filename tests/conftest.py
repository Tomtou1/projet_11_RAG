from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import SpacyTextSplitter
from src.read_input_data import read_and_process_inputdata
import pytest
from dotenv import load_dotenv

load_dotenv()

@pytest.fixture(scope="session")
def processed_docs():
    print("Loading testing fixtures...")
    file_path = "data/evenements-publics-openagenda-full.json"
    processed_docs = read_and_process_inputdata(file_path, debug=False)
    return processed_docs

@pytest.fixture(scope="session")
def chunks(processed_docs):
    nlp_splitter = SpacyTextSplitter(chunk_size=4000, pipeline="fr_core_news_sm")
    chunks = nlp_splitter.split_documents(processed_docs)
    return chunks

@pytest.fixture(scope="session")
def vector_store():
    # Charge l'index FAISS pour l'ensemble des tests
    index_path = "vectorstore/faiss_openagenda_index"
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
    return vector_store