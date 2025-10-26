from flask import Flask, request, jsonify
app = Flask(__name__)

@app.route('/parse', methods=['POST'])
def parse():
    data = request.get_json()
    msg = data.get("message", "").lower()
    if "go to" in msg or "navigate" in msg:
        return jsonify({"intent": "navigation", "entity": "ECE Block"})
    return jsonify({"intent": "general", "answer": "This is a mock response from NLP model."})

if __name__ == "__main__":
    app.run(port=5000)
