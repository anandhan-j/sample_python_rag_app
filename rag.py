import faiss
import pickle
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

def ask_rag(question, k=3):
    q_emb = embedder.encode([question])

    distances, indices = index.search(q_emb, k)

    context = "\n\n".join([texts[i] for i in indices[0]])

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
