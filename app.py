from flask import Flask, render_template, request
import wikipedia
import nltk
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# ----------------- Setup -----------------
nltk.download('punkt')
from nltk.tokenize import sent_tokenize

# Load lightweight NLI model
tokenizer = AutoTokenizer.from_pretrained("cross-encoder/nli-distilroberta-base")
model = AutoModelForSequenceClassification.from_pretrained("cross-encoder/nli-distilroberta-base")
device = torch.device("cpu")
model.to(device)

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
    """Verify claim using lightweight NLI model"""
    if not evidence:
        return "Not Enough Evidence"
    inputs = tokenizer(claim, evidence, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        logits = model(**inputs).logits.to(device)
        probs = torch.softmax(logits, dim=-1)[0]
        label = torch.argmax(probs).item()
        mapping = {0: "Refuted", 1: "Not Enough Evidence", 2: "Supported"}
        return mapping[label]

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
