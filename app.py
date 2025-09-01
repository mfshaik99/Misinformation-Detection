from flask import Flask, render_template, request
import os
import wikipedia
import spacy
import nltk
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import subprocess

# Download NLTK punkt tokenizer
nltk.download('punkt')
from nltk.tokenize import sent_tokenize

# Load SpaCy model safely
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")

# Load Transformers model
tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-mnli")
model = AutoModelForSequenceClassification.from_pretrained("facebook/bart-large-mnli")

app = Flask(__name__)

# ----------------- Helper Functions -----------------
def extract_claims(text):
    """Extract sentences with more than 5 words (simple heuristic)"""
    sents = sent_tokenize(text)
    return [s for s in sents if len(s.split()) > 5][:10]

def get_wiki_content(claim):
    """Retrieve content from Wikipedia"""
    try:
        titles = wikipedia.search(claim, results=3)
        content = ""
        for t in titles:
            page = wikipedia.page(t, auto_suggest=False)
            content += page.content
        return content
    except:
        return ""

def verify_claim(claim, evidence):
    """Verify claim using Transformers NLI model"""
    if not evidence:
        return "Not Enough Evidence"
    inputs = tokenizer(claim, evidence, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[0]
        label = torch.argmax(probs).item()
        mapping = {0: "Refuted", 1: "Not Enough Evidence", 2: "Supported"}
        return mapping[label]

# ----------------- Flask Routes -----------------
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

# ----------------- Main -----------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
