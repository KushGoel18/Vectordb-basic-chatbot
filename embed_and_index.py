# embed_and_index.py

import os
import pickle
import numpy as np
import faiss
from docx import Document
from sentence_transformers import SentenceTransformer

DATA_DIR = "data/"
INDEX_PATH = "vector_store/faiss_index.faiss"
PICKLE_PATH = "vector_store/metadata.pkl"  # Contains list of dicts

def chunk_text(text, max_words=100, overlap=20):
    words = text.split()
    chunks = []
    for i in range(0, len(words), max_words - overlap):
        chunk = " ".join(words[i:i + max_words])
        chunks.append(chunk)
    return chunks

def load_all_docx(data_folder):
    all_chunks = []
    for filename in os.listdir(data_folder):
        if filename.endswith(".docx"):
            path = os.path.join(data_folder, filename)
            doc = Document(path)
            text = "\n".join([para.text for para in doc.paragraphs if para.text.strip()])
            chunks = chunk_text(text)
            for chunk in chunks:
                all_chunks.append({"text": chunk, "source": filename})
            print(f"✅ Loaded {filename} with {len(chunks)} chunks")
    return all_chunks

def embed_and_store():
    all_chunks = load_all_docx(DATA_DIR)

    model = SentenceTransformer('all-MiniLM-L6-v2')

    # 🔧 Normalize embeddings for cosine similarity
    embeddings = model.encode([chunk["text"] for chunk in all_chunks], normalize_embeddings=True)

    dim = embeddings.shape[1]

    # 🔧 Use inner product index (cosine similarity when normalized)
    index = faiss.IndexFlatIP(dim)
    index.add(np.array(embeddings))

    os.makedirs("vector_store", exist_ok=True)
    faiss.write_index(index, INDEX_PATH)

    with open(PICKLE_PATH, 'wb') as f:
        pickle.dump(all_chunks, f)

    print(f"✅ Indexed {len(all_chunks)} chunks into FAISS (cosine similarity).")

if __name__ == "__main__":
    embed_and_store()
