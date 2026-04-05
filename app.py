from flask import Flask, request, jsonify
import json
import pickle
import re


app = Flask(__name__)

with open("model/intent_model.pkl", "rb") as f:
    model, vectorizer = pickle.load(f)

with open("data/intents.json", encoding="utf-8") as f:
    intents = json.load(f)["intents"]


INTENT_BY_TAG = {intent["tag"]: intent for intent in intents}
NAVIGATION_CUES = (
    "navigate",
    "guide me",
    "take me",
    "show me",
    "route me",
    "bring me",
    "way to",
    "go to",
    "get to",
    "reach",
    "where is",
    "how do i get",
)

DESTINATION_ALIASES = {
    "ece": "ECE Block",
    "ece block": "ECE Block",
    "electronics": "ECE Block",
    "cse": "CSE Department",
    "cse block": "CSE Department",
    "computer science": "CSE Department",
    "computer science department": "CSE Department",
    "front gate": "SIT Front Gate",
    "gate": "SIT Front Gate",
    "mba": "MBA Block",
    "mba block": "MBA Block",
    "shiv temple": "SIT Shiv Temple",
    "temple": "SIT Shiv Temple",
    "basketball court": "SIT Basketball Court",
    "basketball": "SIT Basketball Court",
    "volleyball court": "SIT Volleyball Court",
    "volleyball": "SIT Volleyball Court",
    "admin block": "Administration Block",
    "administration": "Administration Block",
    "administration block": "Administration Block",
    "board room": "Board Room",
    "library": "SIT Library",
    "sit library": "SIT Library",
    "annexe": "SIT Annexe",
    "sit annexe": "SIT Annexe",
    "tennis court": "SIT Tennis Court",
    "physics and chemistry": "Physics And Chemistry Department",
    "physics chemistry": "Physics And Chemistry Department",
    "physics and chemistry department": "Physics And Chemistry Department",
    "civil": "Civil Department",
    "civil department": "Civil Department",
    "lbs": "LBS Block",
    "lbs block": "LBS Block",
    "bvb": "BVB Block",
    "bvb block": "BVB Block",
    "alp": "ALP Block",
    "alp block": "ALP Block",
    "canteen": "SIT Canteen",
    "sit canteen": "SIT Canteen",
}


def normalize_text(text):
    lowered = text.lower()
    cleaned = re.sub(r"[^a-z0-9\s]", " ", lowered)
    return re.sub(r"\s+", " ", cleaned).strip()


def build_navigation_lookup():
    lookup = {}
    for intent in intents:
        if intent["tag"].startswith("navigation_request"):
            entity = intent.get("entity", [])
            if entity:
                lookup[entity[0]] = intent
    return lookup


NAVIGATION_BY_DESTINATION = build_navigation_lookup()


def detect_destination(user_input):
    normalized = normalize_text(user_input)
    if not normalized:
        return None

    matches = []

    for alias, destination in DESTINATION_ALIASES.items():
        alias_normalized = normalize_text(alias)
        if alias_normalized in normalized:
            matches.append((len(alias_normalized), destination))

    for destination in NAVIGATION_BY_DESTINATION:
        destination_normalized = normalize_text(destination)
        if destination_normalized in normalized:
            matches.append((len(destination_normalized), destination))

    if not matches:
        return None

    return max(matches, key=lambda item: item[0])[1]


def is_navigation_request(user_input):
    normalized = normalize_text(user_input)
    return any(cue in normalized for cue in NAVIGATION_CUES)


def build_response(intent):
    intent_tag = intent["tag"]
    intent_type = "navigation_request" if intent_tag.startswith("navigation_request") else intent_tag
    entity = intent.get("entity", [])
    return jsonify(
        {
            "intent": intent_tag,
            "intent_type": intent_type,
            "response": intent["responses"][0],
            "entity": entity,
            "destination": entity[0] if entity else None,
        }
    )


def get_response(user_input):
    destination = detect_destination(user_input)
    if destination and is_navigation_request(user_input):
        navigation_intent = NAVIGATION_BY_DESTINATION.get(destination)
        if navigation_intent:
            return build_response(navigation_intent)

    X = vectorizer.transform([user_input])
    if X.nnz == 0:
        fallback = INTENT_BY_TAG.get("fallback")
        if fallback:
            return build_response(fallback)
        return jsonify({"intent": "unknown", "response": "Sorry, I didn't understand that."})

    intent_tag = str(model.predict(X)[0])

    if destination and intent_tag.startswith("navigation_request"):
        navigation_intent = NAVIGATION_BY_DESTINATION.get(destination)
        if navigation_intent:
            return build_response(navigation_intent)

    intent = INTENT_BY_TAG.get(intent_tag)
    if intent:
        return build_response(intent)

    return jsonify({"intent": "unknown", "response": "Sorry, I didn't understand that."})


@app.get("/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/parse", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}
    user_input = data.get("message", "")
    return get_response(user_input)


if __name__ == "__main__":
    app.run(debug=True)
