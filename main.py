"""
main.py — Flask app wiring the AI assistant + RAG pipeline.

Routes:
  GET  /           — serve index.html
  POST /ask        — general Q&A via Groq
  POST /summarize  — text summarization via Groq
  POST /upload     — upload & index a PDF (max 10 MB)
  POST /rag-ask    — citation-grounded answer from indexed papers
  GET  /papers     — list all indexed paper filenames
  POST /remove     — remove a paper from the index
  GET  /status     — returns current chunk count
"""

import os
import tempfile

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request
from groq import Groq

from rag_pipeline import RAGPipeline, MAX_PDF_MB

app = Flask(__name__)
load_dotenv()

# ── Clients ───────────────────────────────────────────────────────────────────

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
rag    = RAGPipeline()

# ── Pages ─────────────────────────────────────────────────────────────────────

@app.route("/")
def hello_world():
    return render_template("index.html")

# ── General routes ────────────────────────────────────────────────────────────

@app.route("/ask", methods=["POST"])
def ask():
    question = request.form.get("question", "").strip()
    if not question:
        return jsonify({"error": "Question is required."}), 400
    if len(question) > 4000:
        return jsonify({"error": "Question is too long. Please limit to 4,000 characters."}), 400

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": "You are a helpful personal assistant. Format your responses clearly using markdown where appropriate."},
            {"role": "user",   "content": question},
        ],
        temperature=0.7,
        max_tokens=512,
    )
    return jsonify({"answer": response.choices[0].message.content.strip()}), 200


@app.route("/summarize", methods=["POST"])
def summarize():
    text = request.form.get("text", "").strip()
    if not text:
        return jsonify({"error": "Text is required."}), 400
    if len(text) > 8000:
        return jsonify({"error": "Text too long. Please limit to ~8,000 characters."}), 400

    prompt   = f"Summarize the following text in 2-3 concise sentences:\n\n{text}"
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": "You are an expert summarizer. Be concise and accurate."},
            {"role": "user",   "content": prompt},
        ],
        temperature=0.5,
        max_tokens=256,
    )
    return jsonify({"summary": response.choices[0].message.content.strip()}), 200

# ── RAG routes ────────────────────────────────────────────────────────────────

@app.route("/upload", methods=["POST"])
def upload():
    if "pdf" not in request.files:
        return jsonify({"error": "No file provided. Use field name 'pdf'."}), 400

    file = request.files["pdf"]

    if not file.filename.lower().endswith(".pdf"):
        return jsonify({"error": "Only PDF files are supported."}), 400

    # Size check before saving to disk
    file.seek(0, 2)
    size_mb = file.tell() / (1024 * 1024)
    file.seek(0)

    if size_mb > MAX_PDF_MB:
        return jsonify({
            "error": f"File is {size_mb:.1f} MB — exceeds the {MAX_PDF_MB} MB limit."
        }), 400

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        file.save(tmp.name)
        tmp_path = tmp.name

    try:
        count = rag.add_pdf(tmp_path, original_filename=file.filename)
        return jsonify({
            "message":      f"Indexed {count} chunks from '{file.filename}' ({size_mb:.1f} MB).",
            "total_chunks": rag.doc_count,
        }), 200
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        os.unlink(tmp_path)


@app.route("/rag-ask", methods=["POST"])
def rag_ask():
    query = (request.form.get("question") or "").strip()
    if not query:
        return jsonify({"error": "Question is required."}), 400
    if len(query) > 2000:
        return jsonify({"error": "Question is too long. Please limit to 2,000 characters."}), 400

    try:
        result = rag.answer(query)
        return jsonify(result), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/papers", methods=["GET"])
def list_papers():
    """Return a list of all currently indexed paper filenames."""
    try:
        papers = rag.get_indexed_papers()
        return jsonify({"papers": papers}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/remove", methods=["POST"])
def remove_paper():
    """Remove all chunks for a given filename from the index."""
    filename = (request.form.get("filename") or "").strip()
    if not filename:
        return jsonify({"error": "Filename is required."}), 400
    try:
        removed = rag.remove_paper(filename)
        if removed == 0:
            return jsonify({"error": f"'{filename}' not found in index."}), 404
        return jsonify({
            "message":      f"Removed '{filename}' ({removed} chunks).",
            "total_chunks": rag.doc_count,
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/status", methods=["GET"])
def status():
    return jsonify({"indexed_chunks": rag.doc_count}), 200

# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(debug=True, threaded=True)
