import os 
from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
import re
import requests
from bs4 import BeautifulSoup


model = SentenceTransformer("all-MiniLM-L6-v2")

chroma_client = chromadb.PersistentClient(path="./shl_db")
collection = chroma_client.get_or_create_collection(name="shl_assessments")

app = Flask(__name__)

def extract_text_from_url(url):
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")
        paragraphs = soup.find_all(["p", "li"])
        text = " ".join(p.get_text() for p in paragraphs)
        return text.strip()
    except Exception as e:
        return f"Unable to extract text from URL: {str(e)}"

@app.route("/recommend", methods=["POST"])
def recommend():
    data = request.json
    query = data.get("query", "")

    if not query:
        return jsonify({"error": "Missing query text"}), 400

    url_match = re.search(r"https?://\S+", query)
    if url_match:
        url = url_match.group(0)
        extracted = extract_text_from_url(url)
        full_query = query + " " + extracted
    else:
        full_query = query

    query_embedding = model.encode(full_query).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3
    )

    recommendations = []
    for metadata in results["metadatas"][0]:
        recommendations.append({
            "Assessment name": metadata.get("Assessment Name", ""),
            "Test Type": metadata.get("Test Type", ""),
            "Duration": metadata.get("Duration", ""),
            "Remote Testing": metadata.get("Remote Testing", ""),
            "URL": metadata.get("URL", "").strip()
        })

    return jsonify({"recommendations": recommendations})

@app.route("/", methods=["GET"])
def home():
    return "✅ SHL Recommendation API is running."


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))  
    app.run(debug=False, host="0.0.0.0", port=port)
