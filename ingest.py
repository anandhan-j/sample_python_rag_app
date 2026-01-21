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
doc_metadata = {}  # Store metadata for each text chunk (index -> metadata)

for file in os.listdir(DATA_DIR):
    if file.endswith(".pdf"):
        reader = PdfReader(os.path.join(DATA_DIR, file))
        for page_num, page in enumerate(reader.pages):
            text = page.extract_text()
            if text and text.strip():  # Only add non-empty pages
                idx = len(texts)
                texts.append(text)
                doc_metadata[idx] = {
                    'source': file,
                    'page': page_num
                }

print(f"Processing {len(texts)} text chunks from {len(set([m['source'] for m in doc_metadata.values()]))} documents")

embeddings = model.encode(texts)

index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

faiss.write_index(index, f"{VECTOR_DIR}/index.faiss")

with open(f"{VECTOR_DIR}/texts.pkl", "wb") as f:
    pickle.dump(texts, f)

with open(f"{VECTOR_DIR}/metadata.pkl", "wb") as f:
    pickle.dump(doc_metadata, f)

print("✅ Documents indexed successfully")
print(f"   - Total chunks: {len(texts)}")
print(f"   - Documents: {len(set([m['source'] for m in doc_metadata.values()]))}")
