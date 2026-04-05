from flask import Flask, request, jsonify
import pickle
import json

app = Flask(__name__)

with open("model/intent_model.pkl", "rb") as f:
    model, vectorizer = pickle.load(f)

with open("data/intents.json") as f:
    intents = json.load(f)["intents"]

def get_response(user_input):
    X = vectorizer.transform([user_input])
    intent_tag = model.predict(X)[0]

    for intent in intents:
        if intent["tag"] == intent_tag:
            intent_type = "navigation_request" if intent_tag.startswith("navigation_request") else intent_tag
            entity = intent.get("entity", [])
            return jsonify({
                "intent": intent_tag,
                "intent_type": intent_type,
                "response": intent["responses"][0],
                "entity": entity,
                "destination": entity[0] if entity else None
            })
    return jsonify({"intent": "unknown", "response": "Sorry, I didn’t understand that."})


@app.route("/parse", methods=["POST"])
def chat():
    data = request.get_json()
    user_input = data.get("message", "")
    return get_response(user_input)


if __name__ == "__main__":
    app.run(debug=True)
