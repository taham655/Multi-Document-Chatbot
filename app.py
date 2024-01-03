import os
import streamlit as st
from ingestionService import IngestionService 

def main():
    st.title("Multi Document Chatbot")

    # Set up the ingestion service
    @st.cache_resource
    def get_ingestion_service():
        documents_directory = "document"  # Specify the path to your documents
        ingestion_service = IngestionService(documents_directory)
        return ingestion_service.retrieve_documents()

    conversation_chain = get_ingestion_service()

    # Initialize chat history in session state if it doesn't exist
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display existing chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User input (question)
    if prompt := st.chat_input("Type your question here."):
        if prompt:  # Check for non-empty prompt
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})

            # Display user message in chat message container
            with st.chat_message("user"):
                st.markdown(prompt)

            # Show a loading animation while retrieving the answer
            with st.spinner("Retrieving the answer..."):
                try:
                    # Retrieve answer from the chatbot
                    # conversation_chain = ingestion_service.retrieve_documents()
                    response = conversation_chain(prompt)
                except Exception as e:
                    response = f"Error in processing your question: {e}"

            # Display assistant response in chat message container
            with st.chat_message("assistant"):
                st.markdown(response['answer'])
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response['answer']})

if __name__ == "__main__":
    main()
