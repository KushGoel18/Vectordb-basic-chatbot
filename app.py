import os
import pickle
import numpy as np
import faiss
from flask import Flask, request, render_template_string
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge import Rouge
import nltk

nltk.download('punkt')

app = Flask(__name__)

# Load FAISS index and metadata
faiss_index = faiss.read_index("vector_store/faiss_index.faiss")
with open("vector_store/metadata.pkl", "rb") as f:
    metadata = pickle.load(f)

# Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head><title>Vector Bot</title></head>
<body>
    <h2>Ask a question:</h2>
    <form method="post">
        <textarea name="query" rows="4" cols="50">{{ query }}</textarea><br><br>
        <input type="submit" value="Submit">
    </form>
    {% if response %}
    <h3>Bot Response:</h3>
    <p>{{ response }}</p>

    <h4>Retrieved Chunks with Scores:</h4>
    <ul>
    {% for chunk, score in retrieved_chunks %}
        <li><strong>{{ chunk.source }}</strong> [{{ "%.2f"|format(score) }}]<br>{{ chunk.text }}</li>
    {% endfor %}
    </ul>
    {% endif %}
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def index():
    query = ""
    response = ""
    retrieved_chunks = []

    if request.method == "POST":
        query = request.form["query"]

        # 🔧 Normalize for cosine similarity search
        query_vector = model.encode([query], normalize_embeddings=True)
        scores, indices = faiss_index.search(np.array(query_vector), k=5)

        top_chunks = [metadata[idx] for idx in indices[0]]
        retrieved_chunks = list(zip(top_chunks, scores[0]))

        context = "\n".join([chunk["text"] for chunk in top_chunks])

        # Bot response
        response = f"Based on the documents:\n{context[:500]}..." if context else "No relevant documents found."

        # Evaluation Metrics
        reference = [query.split()]
        candidate = context.split()

        bleu_score = sentence_bleu(reference, candidate, smoothing_function=SmoothingFunction().method1)
        print(f"\n📊 BLEU Score: {bleu_score:.4f}")

        rouge = Rouge()
        try:
            rouge_scores = rouge.get_scores(context, query)
            print(f"📊 ROUGE Scores: {rouge_scores}")
        except:
            print("📊 ROUGE failed due to empty context or bad input.")

        # 🔧 Normalize again for cosine similarity printout
        context_vector = model.encode([context], normalize_embeddings=True)
        query_vector_cos = model.encode([query], normalize_embeddings=True)
        cosine = cosine_similarity(context_vector, query_vector_cos)[0][0]
        print(f"📊 Cosine Similarity: {cosine:.4f}\n")
    return render_template_string(HTML_TEMPLATE, query=query, response=response, retrieved_chunks=retrieved_chunks)

if __name__ == "__main__":
    app.run(debug=True)
