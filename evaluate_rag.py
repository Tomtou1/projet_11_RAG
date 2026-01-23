import json
from langchain_huggingface import HuggingFaceEmbeddings
import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics import Faithfulness, AnswerRelevancy, ContextRecall, AnswerCorrectness
from langchain_mistralai.chat_models import ChatMistralAI
from src.create_agent import initialize_rag_system


with open("evaluation/dataset_eval_2.json", 'r', encoding='utf-8') as f:
    eval_questions = json.load(f)

# Initialize the agent
agent, vector_store = initialize_rag_system(has_streamlit=False)

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
evaluator_embeddings = LangchainEmbeddingsWrapper(embeddings)

score = evaluate(
    eval_dataset, 
    metrics=[Faithfulness(), AnswerRelevancy(), ContextRecall(), AnswerCorrectness()], 
    llm=evaluator_llm,
    embeddings=evaluator_embeddings
)

score_df = score.to_pandas()
print(score_df[['Faithfulness', 'AnswerRelevancy', 'ContextRecall', 'AnswerCorrectness']])