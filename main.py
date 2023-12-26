import streamlit as st
from app import create_vector, QA_retriever

st.title("Llama2 Q&A chatbot 🤖 ")
btn = st.button("Restore the data")    
if btn:
    create_vector()

question = st.text_input("Question: ")

if question:
    chain = QA_retriever()
    response = chain(question)

    st.header("Answer")
    st.write(response["result"])

