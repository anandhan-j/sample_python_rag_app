import faiss
import pickle
import os
from sentence_transformers import SentenceTransformer
from openai import OpenAI

VECTOR_DIR = "vectorstore"

embedder = SentenceTransformer("all-MiniLM-L6-v2")

client = OpenAI(
    base_url="http://192.168.1.34:1234/v1",
    api_key="lm-studio"
)

index = faiss.read_index(f"{VECTOR_DIR}/index.faiss")

with open(f"{VECTOR_DIR}/texts.pkl", "rb") as f:
    texts = pickle.load(f)

# Load metadata if it exists, otherwise create default metadata
metadata_file = f"{VECTOR_DIR}/metadata.pkl"
if os.path.exists(metadata_file):
    with open(metadata_file, "rb") as f:
        doc_metadata = pickle.load(f)
else:
    # Create default metadata for backward compatibility
    doc_metadata = {i: {'source': 'unknown', 'page': 0} for i in range(len(texts))}

def get_context(question, k=3, selected_docs=None):
    """Retrieve relevant context for a question, optionally filtering by selected documents"""
    q_emb = embedder.encode([question])
    distances, indices = index.search(q_emb, k * 3)  # Get more results to filter
    
    # Filter by selected documents if specified
    if selected_docs:
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
            context = "\n\n".join([texts[i] for i in indices[0][:k]])
    else:
        context = "\n\n".join([texts[i] for i in indices[0][:k]])
    
    return context

def ask_rag(question, k=3):
    """Non-streaming version of RAG query"""
    context = get_context(question, k)
    
    prompt = f"""
You are a document assistant.
Answer ONLY using the context below.
If the answer is not in the context, say:
"Not found in documents."

Context:
{context}

Question:
{question}
"""

    response = client.chat.completions.create(
        model="local-model",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.2
    )

    return response.choices[0].message.content.strip()

def ask_rag_stream(question, k=3):
    """Streaming version of RAG query - yields chunks of response"""
    context = get_context(question, k)
    
    prompt = f"""
You are a document assistant.
Answer ONLY using the context below.
If the answer is not in the context, say:
"Not found in documents."

Context:
{context}

Question:
{question}
"""

    response = client.chat.completions.create(
        model="local-model",
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
    context = get_context(question, k, selected_docs)
    
    prompt = f"""
You are a document assistant.
Answer ONLY using the context below.
If the answer is not in the context, say:
"Not found in documents."

Context:
{context}

Question:
{question}
"""

    response = client.chat.completions.create(
        model="local-model",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
        stream=True
    )

    for chunk in response:
        if chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content
