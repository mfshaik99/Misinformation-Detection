from flask import Flask, render_template, request
import wikipedia
import nltk
import requests
import json

# ----------------- Setup -----------------
nltk.download('punkt')
from nltk.tokenize import sent_tokenize

# Hugging Face Inference API
HF_API_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-mnli"
HF_API_TOKEN = "YOUR_HUGGINGFACE_API_TOKEN"  # replace with your token
headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}

# ----------------- Helper Functions -----------------
def extract_claims(text):
    """Extract sentences with more than 5 words"""
    sents = sent_tokenize(text)
    return [s for s in sents if len(s.split()) > 5][:10]

def get_wiki_content(claim):
    """Retrieve content from Wikipedia (first page, limited to 500 chars)"""
    try:
        titles = wikipedia.search(claim, results=1)
        content = ""
        for t in titles:
            page = wikipedia.page(t, auto_suggest=False)
            content += page.content[:500]  # short snippet to save memory
        return content
    except:
        return ""

def verify_claim(claim, evidence):
    """Verify claim using Hugging Face Inference API"""
    if not evidence:
        return "Not Enough Evidence"
    payload = {
        "inputs": {
            "sequence": evidence,
            "hypothesis": claim
        }
    }
    try:
        response = requests.post(HF_API_URL, headers=headers, json=payload, timeout=30)
        output = response.json()
        if isinstance(output, list) and "label" in output[0]:
            label_map = {"CONTRADICTION": "Refuted", "NEUTRAL": "Not Enough Evidence", "ENTAILMENT": "Supported"}
            return label_map.get(output[0]["label"], "Not Enough Evidence")
        else:
            return "Not Enough Evidence"
    except:
        return "Not Enough Evidence"

# ----------------- Flask App -----------------
app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    claims, verdicts, text = [], [], ""
    if request.method == "POST":
        text = request.form.get("text")
        if text:
            claims = extract_claims(text)
            for c in claims:
                evidence = get_wiki_content(c)
                verdicts.append(verify_claim(c, evidence))
    return render_template("index.html", claims=claims, verdicts=verdicts, text=text)

# ----------------- Run App -----------------
if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))  # Render-assigned port
    app.run(host="0.0.0.0", port=port)
