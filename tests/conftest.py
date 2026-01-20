from langchain_mistralai import MistralAIEmbeddings
from langchain_community.vectorstores import FAISS
import pytest

@pytest.fixture(scope="session")
def vector_store():
    # Charge l'index FAISS pour l'ensemble des tests
    index_path = "vectorstore/faiss_openagenda_index"
    embeddings = MistralAIEmbeddings(model="mistral-embed")
    vector_store = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
    return vector_store


@pytest.fixture(scope="session")
def file_path():
    file_path = "data/evenements-publics-openagenda.json"
    return file_path


