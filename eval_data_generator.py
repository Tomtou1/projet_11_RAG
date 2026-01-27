from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from ragas.testset import TestsetGenerator
from ragas.run_config import RunConfig
from langchain_mistralai import ChatMistralAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.rate_limiters import InMemoryRateLimiter
from src.read_input_data import read_and_process_inputdata


# Model and embeddings initialization
load_dotenv()
rate_limiter = InMemoryRateLimiter(
    requests_per_second=1,  
    check_every_n_seconds=1,
    max_bucket_size=1
)

# Initialize models and embeddings
mistral_llm = ChatMistralAI(model="mistral-large-latest", rate_limiter=rate_limiter)
openai_llm = ChatOpenAI(model="gpt-4o-mini-2024-07-18", rate_limiter=rate_limiter) #gpt-4o-mini-2024-07-18
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Generator and docs initialization
generator = TestsetGenerator.from_langchain(
    llm=openai_llm,
    embedding_model=embedding_model
)

documents = read_and_process_inputdata('data/evenements-publics-openagenda-full.json', debug=True)

print(f"\nDocuments chargés: {len(documents)}")
print(f"Exemple du 1er document:")
print(f"  - Longueur du doc: {len(documents[0].page_content)}")
print(f"  - Metadata: {documents[0].metadata}")
print(f"  - Contenu : {documents[0].page_content[:200]}...")  

# Testset creation and saving
testset = generator.generate_with_langchain_docs(
    documents[:100], 
    testset_size=20, 
    with_debugging_logs=True, 
    run_config=RunConfig(max_workers=1)
)

testset_pd = testset.to_pandas()

# Print and save:
print(testset_pd.head())
print(f"\nNombre total d'échantillons générés: {len(testset_pd)}")

testset.to_jsonl("evaluation/dataset_eval_ai.jsonl")
print(f"\nJeu de test sauvegardé dans 'evaluation/dataset_eval_ai.jsonl'")