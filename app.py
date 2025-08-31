from flask import Flask, request, jsonify, render_template
from transformers import pipeline
import sqlite3

app = Flask(__name__)

# NLP model for misinformation detection
classifier = pipeline("text-classification", model="mrm8488/bert-tiny-finetuned-fake-news")

# Initialize SQLite database
conn = sqlite3.connect('data.db', check_same_thread=False)
c = conn.cursor()
c.execute('''CREATE TABLE IF NOT EXISTS queries (id INTEGER PRIMARY KEY, text TEXT, result TEXT)''')
conn.commit()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/check', methods=['POST'])
def check():
    text = request.form['text']
    result = classifier(text)[0]
    
    # Store in database
    c.execute("INSERT INTO queries (text, result) VALUES (?, ?)", (text, result['label']))
    conn.commit()
    
    # TODO: Google Gemini integration can be added here
    
    return jsonify({"label": result['label'], "score": result['score']})

if __name__ == "__main__":
    app.run(debug=True)
