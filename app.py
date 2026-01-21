import streamlit as st
import os
from rag import ask_rag

st.set_page_config(page_title="RAG Document QA", layout="wide")

st.title("📄 RAG Document Question Answering")

uploaded_files = st.file_uploader(
    "Upload PDF documents",
    type=["pdf"],
    accept_multiple_files=True
)

if uploaded_files:
    os.makedirs("data", exist_ok=True)
    for file in uploaded_files:
        with open(f"data/{file.name}", "wb") as f:
            f.write(file.read())

    st.success("Files uploaded. Run ingest.py to index them.")

st.divider()

question = st.text_input("Ask a question from your documents")

if st.button("Ask"):
    if question:
        with st.spinner("Thinking..."):
            answer = ask_rag(question)
        st.subheader("Answer")
        st.write(answer)
