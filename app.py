import gradio as gr
import wikipedia
import spacy
import nltk
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# ----------------- Setup -----------------
# Download NLTK tokenizer
nltk.download('punkt')
from nltk.tokenize import sent_tokenize

# Load SpaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    import subprocess
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")

# Load smaller NLI model to save memory
tokenizer = AutoTokenizer.from_pretrained("valhalla/distilbart-mnli-12-1")
model = AutoModelForSequenceClassification.from_pretrained("valhalla/distilbart-mnli-12-1")
device = torch.device("cpu")
model.to(device)

# ----------------- Helper Functions -----------------
def extract_claims(text):
    """Extract sentences with more than 5 words"""
    sents = sent_tokenize(text)
    return [s for s in sents if len(s.split()) > 5][:10]

def get_wiki_content(claim):
    """Retrieve content from Wikipedia"""
    try:
        titles = wikipedia.search(claim, results=1)  # only 1 page to save memory
        content = ""
        for t in titles:
            page = wikipedia.page(t, auto_suggest=False)
            content += page.content
        return content
    except:
        return ""

def verify_claim(claim, evidence):
    """Verify claim using NLI model"""
    if not evidence:
        return "Not Enough Evidence"
    inputs = tokenizer(claim, evidence, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[0]
        label = torch.argmax(probs).item()
        mapping = {0: "Refuted", 1: "Not Enough Evidence", 2: "Supported"}
        return mapping[label]

def analyze_text(text):
    """Main function for Gradio interface"""
    claims = extract_claims(text)
    verdicts = []
    for c in claims:
        evidence = get_wiki_content(c)
        verdicts.append(verify_claim(c, evidence))
    return dict(zip(claims, verdicts))

# ----------------- Gradio Interface -----------------
iface = gr.Interface(
    fn=analyze_text,
    inputs=gr.Textbox(lines=10, placeholder="Paste text here..."),
    outputs=gr.Label(num_top_classes=10),
    title="AI-Powered Misinformation Checker",
    description="Paste text to extract claims and verify them using AI and Wikipedia."
)

# Launch the interface
iface.launch()
