import re
import json
import requests
import os
import fitz  # PyMuPDF for PDF processing
from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import pipeline  # Load personal trained model

# Initialize Flask app and enable CORS
app = Flask(__name__)
CORS(app)

# Set up upload folder
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Load personal fine-tuned summarization model (replace "my_summarizer" with your model path)
summarizer = pipeline("summarization", model="my_summarizer")

# Ollama API configuration for DeepSeek-R1
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "deepseek-r1:1.5b"  # Adjust to match your locally running model

# Global variable to store extracted PDF text
pdf_text = ""

# Helper Functions
def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file using PyMuPDF."""
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text("text") + "\n"
    return text.strip()

def clean_deepseek_response(response_text):
    """Remove '<think>...</think>' blocks from DeepSeek's response."""
    response_text = re.sub(r"<think>.*?</think>", "", response_text, flags=re.DOTALL)
    return response_text.strip()

# Routes
@app.route("/")
def home():
    """Simple endpoint to check if the API is running."""
    return jsonify({"message": "PDF Summarizer & Q&A API is running!"})

@app.route("/summarize", methods=["POST"])
def summarize_pdf():
    """Summarize the uploaded PDF using the personal summarization model."""
    global pdf_text
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    # Save the uploaded file
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(file_path)

    # Extract text and store it globally
    pdf_text = extract_text_from_pdf(file_path)
    if len(pdf_text.strip()) == 0:
        return jsonify({"error": "No text found in PDF"}), 400

    # Generate summary using the personal model
    summary = summarizer(pdf_text[:1024], max_length=150, min_length=50, do_sample=False)
    return jsonify({"summary": summary[0]["summary_text"]})

@app.route("/ask", methods=["POST"])
def ask_question():
    """Answer questions about the PDF using DeepSeek-R1 via Ollama."""
    global pdf_text
    if not pdf_text:
        return jsonify({"error": "No document uploaded yet. Please upload a PDF via /summarize first."}), 400

    # Get the question from the request
    data = request.json
    question = data.get("question", "")
    if not question:
        return jsonify({"error": "No question provided"}), 400

    # Construct prompt with context and question
    prompt = (
        f"Based on the following context, provide a concise answer to the question in one sentence.\n\n"
        f"Context: {pdf_text[:4000]}\n\n"
        f"Question: {question}\n\n"
        f"Answer:"
    )

    # Send request to Ollama API with streaming enabled
    response = requests.post(OLLAMA_URL, json={"model": MODEL_NAME, "prompt": prompt}, stream=True)

    # Collect the streamed response
    answer_text = ""
    for line in response.iter_lines():
        if line:
            data = line.decode("utf-8")
            json_data = json.loads(data)
            answer_text += json_data.get("response", "")

    # Clean the response to remove thinking tags
    filtered_answer = clean_deepseek_response(answer_text)

    if filtered_answer.strip():
        return jsonify({"answer": filtered_answer.strip()})
    else:
        return jsonify({"error": "DeepSeek did not generate an answer"}), 500

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)
