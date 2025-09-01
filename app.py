from flask import Flask, render_template, request
import wikipedia
import spacy
import nltk
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

nltk.download('punkt')
from nltk.tokenize import sent_tokenize

# Initialize models
nlp = spacy.load("en_core_web_sm")
tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-mnli")
model = AutoModelForSequenceClassification.from_pretrained("facebook/bart-large-mnli")

app = Flask(__name__)

# Helper: extract claims
def extract_claims(text):
    sents = sent_tokenize(text)
    claims = []
    for s in sents:
        if len(s.split()) > 5:  # simple heuristic
            claims.append(s)
    return claims[:10]

# Helper: retrieve Wikipedia content
def get_wiki_content(claim):
    try:
        titles = wikipedia.search(claim, results=3)
        content = ""
        for t in titles:
            page = wikipedia.page(t, auto_suggest=False)
            content += page.content
        return content
    except:
        return ""

# Helper: verify claim using NLI
def verify_claim(claim, evidence):
    if not evidence:
        return "Not Enough Evidence"
    inputs = tokenizer(claim, evidence, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[0]
        # labels: CONTRADICTION, NEUTRAL, ENTAILMENT
        label = torch.argmax(probs).item()
        mapping = {0: "Refuted", 1: "Not Enough Evidence", 2: "Supported"}
        return mapping[label]

# Routes
@app.route("/", methods=["GET", "POST"])
def index():
    claims = []
    verdicts = []
    text = ""
    if request.method == "POST":
        text = request.form.get("text")
        if text:
            claims = extract_claims(text)
            for c in claims:
                evidence = get_wiki_content(c)
                verdict = verify_claim(c, evidence)
                verdicts.append(verdict)
    return render_template("index.html", claims=claims, verdicts=verdicts, text=text)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
