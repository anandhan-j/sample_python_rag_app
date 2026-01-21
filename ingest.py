import os
import faiss
import pickle
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

DATA_DIR = "data"
VECTOR_DIR = "vectorstore"

os.makedirs(VECTOR_DIR, exist_ok=True)

model = SentenceTransformer("all-MiniLM-L6-v2")
# model = SentenceTransformer("nomic-embed-text-v1.5")

texts = []

for file in os.listdir(DATA_DIR):
    if file.endswith(".pdf"):
        reader = PdfReader(os.path.join(DATA_DIR, file))
        for page in reader.pages:
            texts.append(page.extract_text())

embeddings = model.encode(texts)

index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

faiss.write_index(index, f"{VECTOR_DIR}/index.faiss")

with open(f"{VECTOR_DIR}/texts.pkl", "wb") as f:
    pickle.dump(texts, f)

print("✅ Documents indexed successfully")
