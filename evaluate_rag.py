import os
import json
import faiss
import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics import Faithfulness, AnswerRelevancy, ContextRecall, AnswerCorrectness
from langchain_mistralai.chat_models import ChatMistralAI
from langchain_mistralai import MistralAIEmbeddings
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain.tools import tool
from langchain.agents import create_agent
from src.read_input_data import read_and_process_inputdata
from src.vectorisation import text_split_and_vectorize


with open("evaluation/dataset_eval_2.json", 'r', encoding='utf-8') as f:
    eval_questions = json.load(f)


# Initialize the RAG system
def initialize_rag_system():
    # Load keys, model, embeddings
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
    else:
        # Read / Process / Split and Vectorize data
        processed_docs = read_and_process_inputdata('data/evenements-publics-openagenda-full.json', debug=False)
        vector_store = text_split_and_vectorize(processed_docs, vector_store, index_path)

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

# Initialize the agent
agent, vector_store = initialize_rag_system()

def get_rag_response(query):
    retrieved_docs = vector_store.similarity_search(query, k=5)
    context = [doc.page_content for doc in retrieved_docs]
    agent_response = agent.invoke({"messages": [{"role": "user", "content": query}]})
    answer = agent_response["messages"][-1].content
    return answer, context

# Collect Results
results_data = []
for item in eval_questions:
    ans, ctx = get_rag_response(item["question"])
    results_data.append({
        "question": item["question"],
        "answer": ans,
        "contexts": ctx,
        "ground_truth": item["ground_truth"]
    })


# Run RAG evaluation
try: 
    print("Results Data:")
    for result in results_data:
        print(result)
except Exception as e:
    print(f"Error printing results_data: {e}")

eval_dataset = Dataset.from_pandas(pd.DataFrame(results_data))

# Initialize Mistral LLM / HuggingFace Embeddings
mistral_llm = ChatMistralAI(model="mistral-small-latest")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Wrap them for Ragas
evaluator_llm = LangchainLLMWrapper(mistral_llm)
evaluator_embeddings = LangchainEmbeddingsWrapper(mistral_embeddings)

score = evaluate(
    eval_dataset, 
    metrics=[Faithfulness(), AnswerRelevancy(), ContextRecall(), AnswerCorrectness()], 
    llm=evaluator_llm,
    embeddings=evaluator_embeddings
)

score_df = score.to_pandas()
print(score_df[['Faithfulness', 'AnswerRelevancy', 'ContextRecall', 'AnswerCorrectness']])