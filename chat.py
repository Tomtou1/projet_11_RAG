import streamlit as st
from src.create_agent import initialize_rag_system

st.title("Assistant Événements Culturels")

# Initialize the RAG system (only once) with Streamlit caching
@st.cache_resource
def get_rag_system():
    return initialize_rag_system()

# Initialize the agent
agent, vector_store = get_rag_system()

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