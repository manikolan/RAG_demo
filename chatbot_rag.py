#import streamlit
import streamlit as st
import os
from dotenv import load_dotenv
import google.generativeai as genai
from pinecone import Pinecone, ServerlessSpec
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

load_dotenv()

st.title("Jira Bug Fix RAG Chatbot completed")

# initialize pinecone database
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

# initialize pinecone database
index_name = os.environ.get("PINECONE_INDEX_NAME")  # change if desired
index = pc.Index(index_name)

genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))

# initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

    st.session_state.messages.append(SystemMessage("You are an assistant for question-answering tasks. "))

# display chat messages from history on app rerun
for message in st.session_state.messages:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.markdown(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(message.content)

# create the bar where we can type messages
prompt = st.chat_input("How are you?")

# did the user submit a prompt?
if prompt:
    with st.chat_message("user"):
        st.markdown(prompt)
        st.session_state.messages.append(HumanMessage(prompt))

    # Embed the prompt using Gemini
    prompt_embedding = genai.embed_content(
        model="gemini-embedding-001",
        content=[prompt]
    )["embedding"][0]

    # Query Pinecone for similar vectors
    response = index.query(
        vector=prompt_embedding,
        top_k=3,
        include_metadata=True
    )

    # Build context from retrieved documents
    docs_text = ""
    for match in response["matches"]:
        docs_text += match["metadata"].get("text", "") + "\n"

    # Create the system prompt
    system_prompt = f"""You are an AI assistant that has information about the document provided below.\
    If the question has nothing to do with the context, give generic answers like a normal AI Assistant. Thank you!
Context: {docs_text}"""

    st.session_state.messages.append(SystemMessage(system_prompt))

    # Use Gemini chat completion
    chat_response = genai.GenerativeModel("gemini-2.5-flash").generate_content(
        [
            {"role": "model", "parts": [system_prompt]},
            {"role": "user", "parts": [prompt]}
        ]
    )
    result = chat_response.text

    with st.chat_message("assistant"):
        st.markdown(result)
        st.session_state.messages.append(AIMessage(result))