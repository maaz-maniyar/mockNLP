from flask import Flask, request, jsonify
import pickle
import json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

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
            return jsonify({
                "intent": intent_tag,
                "response": intent["responses"][0],
                "entity": intent["entity"]
            })
    return jsonify({"intent": "unknown", "response": "Sorry, I didn’t understand that."})


@app.route("/parse", methods=["POST"])
def chat():
    data = request.get_json()
    user_input = data.get("message", "")
    return get_response(user_input)


if __name__ == "__main__":
    app.run(debug=True)