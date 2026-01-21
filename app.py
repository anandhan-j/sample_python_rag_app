import streamlit as st
import os
import glob
from rag import ask_rag, ask_rag_stream, ask_rag_with_docs

st.set_page_config(page_title="RAG Document QA", layout="wide")

st.title("📄 RAG Document Question Answering")

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

if "streaming" not in st.session_state:
    st.session_state.streaming = True

if "selected_docs" not in st.session_state:
    st.session_state.selected_docs = []

if "processing" not in st.session_state:
    st.session_state.processing = False

# Sidebar for settings
with st.sidebar:
    st.header("⚙️ Settings")
    
    # Streaming toggle
    st.session_state.streaming = st.toggle(
        "Enable Streaming",
        value=st.session_state.streaming,
        help="Display responses in real-time as they're generated"
    )
    
    # Clear chat button
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()
    
    st.divider()
    
    # Upload section
    st.header("📁 Upload Documents")
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
        
        st.success(f"Uploaded {len(uploaded_files)} file(s)")
    
    st.divider()
    
    # Document list and selection
    st.header("📚 Document List")
    
    # Get all PDF files from data directory
    if os.path.exists("data"):
        pdf_files = [f for f in os.listdir("data") if f.lower().endswith('.pdf')]
        
        if pdf_files:
            # Multi-select for documents
            selected = st.multiselect(
                "Select documents to query",
                pdf_files,
                default=st.session_state.selected_docs if st.session_state.selected_docs else []
            )
            st.session_state.selected_docs = selected
            
            if selected:
                st.info(f"Selected: {len(selected)} document(s)")
            else:
                st.warning("No documents selected. Answer will use all indexed documents.")
        else:
            st.info("No PDF files found in data directory")
    else:
        st.info("No data directory found")
    
    st.divider()
    
    # Vector conversion section
    st.header("🔄 Vector Conversion")
    
    if st.button("Convert to Vectors", disabled=st.session_state.processing):
        st.session_state.processing = True
        st.rerun()
    
    if st.session_state.processing:
        with st.spinner("Processing documents..."):
            # Simulate progress
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Step 1: Check for PDF files
            status_text.text("Step 1: Checking for PDF files...")
            progress_bar.progress(20)
            
            if not os.path.exists("data"):
                st.error("No data directory found. Please upload documents first.")
                st.session_state.processing = False
                st.rerun()
            
            pdf_files = [f for f in os.listdir("data") if f.lower().endswith('.pdf')]
            if not pdf_files:
                st.error("No PDF files found in data directory. Please upload documents first.")
                st.session_state.processing = False
                st.rerun()
            
            # Step 2: Run ingest.py
            status_text.text("Step 2: Extracting text from PDFs...")
            progress_bar.progress(40)
            
            try:
                import subprocess
                result = subprocess.run(
                    ["python", "ingest.py"],
                    capture_output=True,
                    text=True,
                    cwd="d:\\Project\\python\\RAG_APP"
                )
                
                if result.returncode == 0:
                    progress_bar.progress(100)
                    status_text.text("✅ Vector conversion completed successfully!")
                    st.success("Documents indexed and vectors created!")
                else:
                    st.error(f"Error during processing: {result.stderr}")
                    status_text.text("❌ Processing failed")
                
            except Exception as e:
                st.error(f"Error running ingest.py: {str(e)}")
                status_text.text("❌ Processing failed")
            
            st.session_state.processing = False
            st.rerun()

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask a question from your documents"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate assistant response
    with st.chat_message("assistant"):
        if st.session_state.streaming:
            # Streaming mode
            response_placeholder = st.empty()
            full_response = ""
            
            # Use selected documents if specified
            if st.session_state.selected_docs:
                for chunk in ask_rag_with_docs(prompt, st.session_state.selected_docs):
                    full_response += chunk
                    response_placeholder.markdown(full_response)
            else:
                for chunk in ask_rag_stream(prompt):
                    full_response += chunk
                    response_placeholder.markdown(full_response)
            
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": full_response})
        else:
            # Non-streaming mode
            with st.spinner("Thinking..."):
                if st.session_state.selected_docs:
                    answer = ask_rag_with_docs(prompt, st.session_state.selected_docs)
                else:
                    answer = ask_rag(prompt)
            st.markdown(answer)
            
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": answer})
