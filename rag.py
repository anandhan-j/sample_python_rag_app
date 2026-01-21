import faiss
import pickle
import os
import numpy as np
from sentence_transformers import SentenceTransformer
from openai import OpenAI

VECTOR_DIR = "vectorstore"

embedder = SentenceTransformer("all-MiniLM-L6-v2")

client = OpenAI(
    base_url="http://192.168.1.34:1234/v1",
    api_key="lm-studio"
)

# Initialize variables with default values
index = None
texts = []
doc_metadata = {}
embeddings = None

# Load index if it exists
index_file = f"{VECTOR_DIR}/index.faiss"
if os.path.exists(index_file):
    index = faiss.read_index(index_file)

# Load texts if they exist
texts_file = f"{VECTOR_DIR}/texts.pkl"
if os.path.exists(texts_file):
    with open(texts_file, "rb") as f:
        texts = pickle.load(f)

# Load metadata if it exists, otherwise create default metadata
metadata_file = f"{VECTOR_DIR}/metadata.pkl"
if os.path.exists(metadata_file):
    with open(metadata_file, "rb") as f:
        doc_metadata = pickle.load(f)
else:
    # Create default metadata for backward compatibility
    doc_metadata = {i: {'source': 'unknown', 'page': 0} for i in range(len(texts))}

# Load embeddings if they exist
embeddings_file = f"{VECTOR_DIR}/embeddings.pkl"
if os.path.exists(embeddings_file):
    with open(embeddings_file, "rb") as f:
        embeddings = pickle.load(f)

def get_context(question, k=3, selected_docs=None):
    """Retrieve relevant context for a question, optionally filtering by selected documents"""
    # Check if index exists
    if index is None:
        return ""
    
    q_emb = embedder.encode([question])
    
    # Filter by selected documents if specified
    if selected_docs:
        # Get indices of chunks that belong to selected documents
        selected_indices = []
        for idx, metadata in doc_metadata.items():
            doc_name = metadata.get('source', '')
            # Check if any selected document name is contained in the metadata source
            # This handles cases where the full path might be stored
            if any(selected in doc_name for selected in selected_docs):
                selected_indices.append(idx)
        
        # Debug logging
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"Selected docs: {selected_docs}")
        logger.info(f"Available doc names in metadata: {set([m.get('source', '') for m in doc_metadata.values()])}")
        logger.info(f"Found {len(selected_indices)} chunks for selected documents")
        
        if not selected_indices:
            # No chunks found for selected documents
            logger.warning(f"No chunks found for selected documents: {selected_docs}")
            return ""
        
        # Search only within selected document chunks
        if embeddings is not None:
            # Use pre-computed embeddings for faster search
            selected_embeddings = []
            for idx in selected_indices:
                selected_embeddings.append(embeddings[idx])
            
            if selected_embeddings:
                selected_embeddings = np.array(selected_embeddings)
                # Create a temporary index for selected embeddings
                temp_index = faiss.IndexFlatL2(selected_embeddings.shape[1])
                temp_index.add(selected_embeddings)
                
                # Search within selected embeddings
                distances, local_indices = temp_index.search(q_emb, min(k, len(selected_indices)))
                
                # Map local indices back to global indices
                filtered_contexts = []
                for local_idx in local_indices[0]:
                    if local_idx < len(selected_indices):
                        global_idx = selected_indices[local_idx]
                        filtered_contexts.append(texts[global_idx])
                        if len(filtered_contexts) >= k:
                            break
                
                if filtered_contexts:
                    context = "\n\n".join(filtered_contexts)
                else:
                    context = ""
            else:
                context = ""
        else:
            # Fallback: search all and filter (slower)
            distances, indices = index.search(q_emb, k * 3)
            filtered_contexts = []
            for i in indices[0]:
                doc_name = doc_metadata.get(i, {}).get('source', '')
                if any(selected in doc_name for selected in selected_docs):
                    filtered_contexts.append(texts[i])
                    if len(filtered_contexts) >= k:
                        break
            
            if filtered_contexts:
                context = "\n\n".join(filtered_contexts)
            else:
                context = ""
    else:
        distances, indices = index.search(q_emb, k)
        context = "\n\n".join([texts[i] for i in indices[0]])
    
    return context

def ask_rag(question, k=3):
    """Non-streaming version of RAG query"""
    # Check if index exists
    if index is None:
        return "No documents have been indexed yet. Please convert documents to vectors first."
    
    context = get_context(question, k)
    
    # Debug logging
    import logging
    logger = logging.getLogger(__name__)
    logger.info(f"Context retrieved: {len(context)} characters")
    
    # Check if context is empty
    if not context:
        return "Not found in documents."
    
    prompt = f"""
You are a document assistant.
Answer ONLY using the context below.
If the answer is not in the context, say:
"Not found in documents."

Context:
{context}

Question:
{question}

IMPORTANT: Only use information from the context above. Do not use any external knowledge or information from other sources.
"""

    response = client.chat.completions.create(
        model="meta-llama-3.1-8b-instruct",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.2
    )

    return response.choices[0].message.content.strip()

def ask_rag_stream(question, k=3):
    """Streaming version of RAG query - yields chunks of response"""
    # Check if index exists
    if index is None:
        yield "No documents have been indexed yet. Please convert documents to vectors first."
        return
    
    context = get_context(question, k)
    
    # Debug logging
    import logging
    logger = logging.getLogger(__name__)
    logger.info(f"Context retrieved: {len(context)} characters")
    
    # Check if context is empty
    if not context:
        yield "Not found in documents."
        return
    
    prompt = f"""
You are a document assistant.
Answer ONLY using the context below.
If the answer is not in the context, say:
"Not found in documents."

Context:
{context}

Question:
{question}

IMPORTANT: Only use information from the context above. Do not use any external knowledge or information from other sources.
"""

    response = client.chat.completions.create(
        model="meta-llama-3.1-8b-instruct",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
        stream=True
    )

    for chunk in response:
        if chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content

def ask_rag_with_docs(question, selected_docs, k=3):
    """Streaming version of RAG query with specific documents - yields chunks of response"""
    # Check if index exists
    if index is None:
        yield "No documents have been indexed yet. Please convert documents to vectors first."
        return
    
    context = get_context(question, k, selected_docs)
    
    # Debug logging
    import logging
    logger = logging.getLogger(__name__)
    logger.info(f"Context retrieved: {len(context)} characters")
    
    # Check if context is empty
    if not context:
        yield "Not found in documents."
        return
    
    prompt = f"""
You are a document assistant.
Answer ONLY using the context below.
If the answer is not in the context, say:
"Not found in documents."

Context:
{context}

Question:
{question}

IMPORTANT: Only use information from the selected documents. Do not use any external knowledge or information from other sources.
"""

    response = client.chat.completions.create(
        model="meta-llama-3.1-8b-instruct",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
        stream=True
    )

    for chunk in response:
        if chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content
