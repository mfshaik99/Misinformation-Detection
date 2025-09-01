from flask import Flask, render_template, request

app = Flask(__name__)

# Simple keyword-based fake news detection
FAKE_KEYWORDS = ["vaccines are unsafe", "chocolate cures cancer", "aliens built pyramids"]

def check_misinformation(text):
    text = text.lower()
    results = []
    sentences = text.split(".")
    for s in sentences:
        s = s.strip()
        if not s:
            continue
        verdict = "Safe"
        for kw in FAKE_KEYWORDS:
            if kw in s:
                verdict = "Possibly False"
                break
        results.append((s, verdict))
    return results

@app.route("/", methods=["GET", "POST"])
def index():
    results = []
    text = ""
    if request.method == "POST":
        text = request.form.get("text", "")
        results = check_misinformation(text)
    return render_template("index.html", results=results, text=text)

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
