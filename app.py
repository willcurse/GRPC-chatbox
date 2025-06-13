import os
import json
import hashlib
import fitz  # PyMuPDF
import faiss
import numpy as np
import pandas as pd

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from dotenv import load_dotenv
from openai import OpenAI

# ======== Setup ========
load_dotenv()
app = Flask(__name__)
CORS(app)

OPENAI_API_KEY = os.getenv("APIKEY")
client = OpenAI(api_key=OPENAI_API_KEY)

# ======== Config ========
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K = 10
FINAL_K = 3
EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o"

DOCUMENT_PATHS = [
    "GRPC 2025 Brochure.pdf",
    "GRPC 2025 CONFERENCE PROGRAM.pdf",
    "Dates.pdf",
    "Contacts.csv",
    "1.pdf",
    "company2025.pdf",
    "General Category.pdf",
    "grpc 2024.pdf",
    "GRPC 2025 Event Details.pdf"
]

# ======== Utils ========
def extract_text_from_pdf(path):
    doc = fitz.open(path)
    return [page.get_text() for page in doc]

def extract_text_from_csv(path):
    try:
        df = pd.read_csv(path)
        return [row for row in df.astype(str).apply(lambda x: ' | '.join(x), axis=1)]
    except Exception as e:
        print(f"Error reading CSV {path}: {e}")
        return []

def extract_text_from_excel(path):
    try:
        df = pd.read_excel(path)
        return [row for row in df.astype(str).apply(lambda x: ' | '.join(x), axis=1)]
    except Exception as e:
        print(f"Error reading Excel {path}: {e}")
        return []

def split_text(text_list, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    chunks = []
    for text in text_list:
        for i in range(0, len(text), chunk_size - overlap):
            chunk = text[i:i + chunk_size].strip()
            if chunk:
                chunks.append(chunk)
    return chunks

def hash_text(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()

def get_embedding(text):
    return client.embeddings.create(model=EMBED_MODEL, input=text).data[0].embedding

# ======== Embedding & Indexing ========
all_chunks = []
metadata = []
embeddings = []
embedding_cache = {}

def load_and_process_documents():
    for path in DOCUMENT_PATHS:
        ext = os.path.splitext(path)[-1].lower()

        if ext == ".pdf":
            pages = extract_text_from_pdf(path)
        elif ext == ".csv":
            pages = extract_text_from_csv(path)
        elif ext in [".xls", ".xlsx"]:
            pages = extract_text_from_excel(path)
        else:
            print(f"Unsupported file type: {path}")
            continue

        chunks = split_text(pages)

        for i, chunk in enumerate(chunks):
            chunk_id = hash_text(chunk)
            all_chunks.append(chunk)
            metadata.append({"source": path, "chunk_index": i})

            if chunk_id in embedding_cache:
                embedding = embedding_cache[chunk_id]
            else:
                embedding = get_embedding(chunk)
                embedding_cache[chunk_id] = embedding
            embeddings.append(embedding)

    return np.array(embeddings).astype("float32")

embedding_matrix = load_and_process_documents()
dimension = len(embeddings[0])
index = faiss.IndexFlatL2(dimension)
index.add(embedding_matrix)

# ======== Routes ========
@app.route("/")
def home():
    return render_template("index.html")

@app.route('/chat', methods=['POST'])
def chat_webhook():
    try:
        raw_data = request.data
        print("Raw request headers:", dict(request.headers))
        print("Raw request body:", raw_data)

        data = json.loads(raw_data.decode('utf-8'))
        print("Parsed data:", data)

        reply_text = "Hi! Your message was received 🎉"

        return jsonify({
            "action": {
                "type": "reply",
                "replies": [
                    {
                        "type": "text",
                        "text": reply_text
                    }
                ]
            }
        }), 200

    except Exception as e:
        print("Error:", e)
        return jsonify({"error": str(e)}), 400

# ======== Internal Chat Function ========
def handle_chat(user_input):
    try:
        query_embed = np.array(get_embedding(user_input)).astype("float32").reshape(1, -1)

        distances, indices = index.search(query_embed, TOP_K)
        retrieved_chunks = [all_chunks[idx] for idx in indices[0]]
        chunk_sources = [metadata[idx] for idx in indices[0]]

        rerank_prompt = f"""
You are a helpful assistant. From the following chunks of context, select the {FINAL_K} most relevant to this question:
"{user_input}"

Context Chunks:
""" + "\n\n".join([f"Chunk {i+1}: {chunk}" for i, chunk in enumerate(retrieved_chunks)])

        rerank_response = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[{"role": "user", "content": rerank_prompt}],
            temperature=0
        )

        refined_context = rerank_response.choices[0].message.content.strip()

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a knowledgeable assistant answering questions based only on the GRPC 2025 documents. "
                    "Respond clearly and professionally in markdown format. Use bullet points, headers, or numbered lists where helpful."
                )
            },
            {
                "role": "user",
                "content": f"### Context:\n{refined_context}\n\n### Question:\n{user_input}"
            }
        ]

        final_response = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=messages,
            temperature=0.2
        )

        return final_response.choices[0].message.content.strip()

    except Exception as e:
        return f"Error processing chat: {e}"

# ======== Main ========
if __name__ == "__main__":
    app.run(debug=True)
