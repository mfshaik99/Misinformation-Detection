import requests  # if Gemini uses HTTP API

# Inside /check route, after Hugging Face classification
# Example Gemini API call (pseudo-code, adapt based on API docs)
gemini_response = requests.post(
    "https://api.google.com/gemini/v1/analyze",
    headers={"Authorization": "Bearer YOUR_GEMINI_API_KEY"},
    json={"text": text}
)
gemini_result = gemini_response.json()
# gemini_result could contain 'verdict', 'sources', 'explanation'

# Include Gemini info in response
return jsonify({
    "label": result['label'],
    "score": result['score'],
    "gemini_verdict": gemini_result.get("verdict", "N/A"),
    "sources": gemini_result.get("sources", []),
    "explanation": gemini_result.get("explanation", "")
})
