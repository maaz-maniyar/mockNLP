from flask import Flask, request, jsonify

app = Flask(__name__)

DESTINATIONS = [
    "SIT Front Gate",
    "Mechanical Department",
    "SIT Coffee Shop",
    "Library",
    "CSE Department",
    "Main Building"
]

@app.route("/parse", methods=["POST"])
def parse():
    data = request.get_json(force=True)
    msg = data.get("message", "").lower()

    for dest in DESTINATIONS:
        if dest.lower() in msg:
            return jsonify({
                "intent": "navigation",
                "entity": dest,
                "answer": f"Navigating to {dest}..."
            })

    if any(x in msg for x in ["hi", "hello", "hey"]):
        return jsonify({
            "intent": "greeting",
            "answer": "Hello! Where would you like to go today?"
        })

    if "who" in msg or "what" in msg:
        return jsonify({
            "intent": "info",
            "answer": "I'm NavSIT, your AR navigation assistant for SIT campus."
        })

    return jsonify({
        "intent": "unknown",
        "answer": "I'm not sure I understood that. Could you repeat?"
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)