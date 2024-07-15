import streamlit as st
from model import get_conversational_chain, user_input
from dotenv import load_dotenv
import os
import google.generativeai as genai




load_dotenv()
api_key = os.getenv('GOOGLE_API_KEY')

os.environ["GOOGLE_API_KEY"] = api_key
genai.configure(api_key = api_key)


model = genai.GenerativeModel(model_name="gemini-pro",
                              )
st.markdown(body= '# APPCLICK SUPPORT CHATBOT')


if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("What is up?"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    response = user_input(prompt)
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
