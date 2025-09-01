from flask import Flask, render_template, request
import wikipedia
import nltk
import requests
import traceback

# ----------------- Setup -----------------
nltk.download('punkt')
from nltk.tokenize import sent_tokenize

HF_API_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-mnli"
HF_API_TOKEN = "YOUR_HUGGINGFACE_API_TOKEN"  # Replace with your valid token
HEADERS = {"Authorization": f"Bearer {HF_API_TOKEN}"}

# ----------------- Helper Functions -----------------
def extract_claims(text):
    """Extract up to 5 sentences with more than 5 words"""
    sents = sent_tokenize(text)
    return [s for s in sents if len(s.split()) > 5][:5]

def get_wiki_content(claim):
    """Retrieve first 500 characters from the first relevant Wikipedia page"""
    try:
        titles = wikipedia.search(claim, results=1)
        if not titles:
            return ""
        page = wikipedia.page(titles[0], auto_suggest=False)
        return page.content[:500]
    except Exception as e:
        print(f"Wikipedia Error for claim '{claim}': {e}")
        return ""

def verify_claim(claim, evidence):
    """Verify claim using Hugging Face API"""
    if not evidence:
        return "Not Enough Evidence"
    payload = {"inputs": {"sequence": evidence, "hypothesis": claim}}
    try:
        response = requests.post(HF_API_URL, headers=HEADERS, json=payload, timeout=60)
        output = response.json()
        if isinstance(output, list) and "label" in output[0]:
            label_map = {"CONTRADICTION": "Refuted", "NEUTRAL": "Not Enough Evidence", "ENTAILMENT": "Supported"}
            return label_map.get(output[0]["label"], "Not Enough Evidence")
        else:
            print("Unexpected HF Response:", output)
            return "Not Enough Evidence"
    except Exception as e:
        print("Hugging Face API Error:", e)
        traceback.print_exc()
        return "Not Enough Evidence"

# ----------------- Flask App -----------------
app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    claims, verdicts, text = [], [], ""
    if request.method == "POST":
        text = request.form.get("text", "")
        if text.strip():
            try:
                claims = extract_claims(text)
                for c in claims:
                    evidence = get_wiki_content(c)
                    verdicts.append(verify_claim(c, evidence))
            except Exception as e:
                print("Processing Error:", e)
                traceback.print_exc()
                claims = ["Error occurred"]
                verdicts = ["Please try again"]
    return render_template("index.html", claims=claims, verdicts=verdicts, text=text)

# ----------------- Run App -----------------
if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))  # Render assigned port
    app.run(host="0.0.0.0", port=port)
