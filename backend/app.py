import re
import json
import requests
import os
import fitz  # PyMuPDF for PDF processing
from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import pipeline  # ✅ Load personal trained model

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# ✅ Load your personal fine-tuned summarization model
summarizer = pipeline("summarization", model="my_summarizer")

# ✅ Ollama API URL for DeepSeek
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "deepseek-r1:1.5b"  # ✅ Matches your locally running model

pdf_text = ""  # Stores extracted PDF text for Q&A

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file."""
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text("text") + "\n"
    return text.strip()

def clean_deepseek_response(response_text):
   # """Removes '<think> ... </think>' blocks and returns only the final answer."""
    response_text = re.sub(r"<think>.*?</think>", "", response_text, flags=re.DOTALL)  # Remove full <think> tags
    return response_text.strip()

@app.route("/")
def home():
    return jsonify({"message": "PDF Summarizer & Q&A API is running!"})

@app.route("/summarize", methods=["POST"])
def summarize_pdf():
    """Summarize the extracted PDF text using Personal Model."""
    global pdf_text
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(file_path)

    pdf_text = extract_text_from_pdf(file_path)
    if len(pdf_text.strip()) == 0:
        return jsonify({"error": "No text found in PDF"}), 400

    # ✅ Use your personal trained summarizer
    summary = summarizer(pdf_text[:1024], max_length=150, min_length=50, do_sample=False)
    return jsonify({"summary": summary[0]["summary_text"]})

@app.route("/ask", methods=["POST"])
def ask_question():
    """Handle Q&A requests using DeepSeek via Ollama."""
    global pdf_text
    if not pdf_text:
        return jsonify({"error": "No document uploaded yet"}), 400

    data = request.json
    question = data.get("question", "")

    if not question:
        return jsonify({"error": "No question provided"}), 400

    # ✅ Use DeepSeek-R1 (via Ollama) for Q&A
    prompt = f"Context: {pdf_text[:4000]}\nQuestion: {question}\nAnswer:"

    response = requests.post(OLLAMA_URL, json={"model": MODEL_NAME, "prompt": prompt}, stream=True)

    # ✅ Collect streamed response & filter out '<think>' blocks
    answer_text = ""
    for line in response.iter_lines():
        if line:
            data = line.decode("utf-8")
            json_data = json.loads(data)
            filtered_response = json_data.get("response", "")
            filtered_response = clean_deepseek_response(filtered_response)
            answer_text += filtered_response + " "

    if answer_text.strip():
        return jsonify({"answer": answer_text.strip()})
    else:
        return jsonify({"error": "DeepSeek did not generate an answer"}), 500

if __name__ == "__main__":
    app.run(debug=True)
