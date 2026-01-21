import streamlit as st
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

st.title("Assistant Événements Culturels")

# Initialize the RAG system (only once)
@st.cache_resource
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
        st.success(f"Loaded existing vector database")
    else:
        st.warning(f"Creating a new vector database...")
        # Read / Process / Split and Vectorize data
        processed_docs = read_and_process_inputdata('data/evenements-publics-openagenda-full.json', debug=False)
        vector_store = text_split_and_vectorize(processed_docs, vector_store, index_path)
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

# Initialize the agent
agent, vector_store = initialize_rag_system()

# Add debug option in sidebar
with st.sidebar:
    st.header("Options")
    show_debug = st.checkbox("Afficher les documents récupérés", value=False)


# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("Posez votre question sur les événements culturels..."):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Get response from RAG agent
    with st.chat_message("assistant"):
        with st.spinner("Recherche en cours..."):
            # Show retrieved documents if debug is enabled
            if show_debug:
                with st.expander("🔍 Documents récupérés"):
                    retrieved_docs = vector_store.similarity_search(prompt, k=5)
                    for i, doc in enumerate(retrieved_docs, 1):
                        st.write(f"**Document {i}:**")
                        st.write(f"Metadata: {doc.metadata}")
                        st.write(f"Content: {doc.page_content[:300]}...")
                        st.divider()
            
            response_text = ""
            # Stream the agent's response
            for event in agent.stream(
                {"messages": [{"role": "user", "content": prompt}]},
                stream_mode="values",
            ):
                last_message = event["messages"][-1]
                # Extract the content from the last message
                if hasattr(last_message, 'content'):
                    response_text = last_message.content
            
            st.markdown(response_text)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response_text})