from flask import Flask, request, jsonify
import json
import logging
import os
import pickle
import re
from threading import Lock


app = Flask(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("mocknlp")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "intent_model.pkl")
INTENTS_PATH = os.path.join(BASE_DIR, "data", "intents.json")

model = None
vectorizer = None
intents = []
INTENT_BY_TAG = {}
NAVIGATION_BY_DESTINATION = {}
MODEL_LOAD_ERROR = None
resource_lock = Lock()

logger.info("Starting MockNLP. cwd=%s, model_path=%s, intents_path=%s", os.getcwd(), MODEL_PATH, INTENTS_PATH)

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


def load_resources():
    global model, vectorizer, intents, INTENT_BY_TAG, NAVIGATION_BY_DESTINATION, MODEL_LOAD_ERROR

    if model is not None and vectorizer is not None and intents:
        return True

    with resource_lock:
        if model is not None and vectorizer is not None and intents:
            return True

        try:
            logger.info("Loading NLP resources from disk.")
            with open(MODEL_PATH, "rb") as f:
                model, vectorizer = pickle.load(f)

            logger.info("Model loaded successfully from %s", MODEL_PATH)

            with open(INTENTS_PATH, encoding="utf-8") as f:
                intents = json.load(f)["intents"]

            INTENT_BY_TAG = {intent["tag"]: intent for intent in intents}
            NAVIGATION_BY_DESTINATION = build_navigation_lookup()
            MODEL_LOAD_ERROR = None
            logger.info("Loaded %s intents from %s", len(intents), INTENTS_PATH)
            return True
        except Exception as error:
            MODEL_LOAD_ERROR = str(error)
            logger.exception("Failed to load NLP resources: %s", error)
            model = None
            vectorizer = None
            intents = []
            INTENT_BY_TAG = {}
            NAVIGATION_BY_DESTINATION = {}
            return False


load_resources()


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
    if not load_resources():
        logger.error("Cannot process message because NLP resources are unavailable. error=%s", MODEL_LOAD_ERROR)
        return jsonify(
            {
                "intent": "service_unavailable",
                "intent_type": "service_unavailable",
                "response": "The NLP model is still loading. Please try again in a moment.",
                "entity": [],
                "destination": None,
            }
        ), 503

    logger.info("Processing message. raw_input=%r", user_input)
    destination = detect_destination(user_input)
    if destination and is_navigation_request(user_input):
        navigation_intent = NAVIGATION_BY_DESTINATION.get(destination)
        if navigation_intent:
            logger.info("Matched navigation via alias/cue. destination=%s, intent=%s", destination, navigation_intent["tag"])
            return build_response(navigation_intent)

    X = vectorizer.transform([user_input])
    if X.nnz == 0:
        fallback = INTENT_BY_TAG.get("fallback")
        if fallback:
            logger.info("No vectorized tokens found. Returning fallback response.")
            return build_response(fallback)
        logger.warning("No vectorized tokens found and fallback intent missing.")
        return jsonify({"intent": "unknown", "response": "Sorry, I didn't understand that."})

    intent_tag = str(model.predict(X)[0])
    logger.info("Model predicted intent=%s, destination=%s", intent_tag, destination)

    if destination and intent_tag.startswith("navigation_request"):
        navigation_intent = NAVIGATION_BY_DESTINATION.get(destination)
        if navigation_intent:
            logger.info("Resolved predicted navigation to destination=%s using intent=%s", destination, navigation_intent["tag"])
            return build_response(navigation_intent)

    intent = INTENT_BY_TAG.get(intent_tag)
    if intent:
        logger.info("Returning response for intent=%s", intent_tag)
        return build_response(intent)

    logger.warning("Predicted intent not found in intents list. intent=%s", intent_tag)
    return jsonify({"intent": "unknown", "response": "Sorry, I didn't understand that."})


@app.get("/health")
def health():
    healthy = load_resources()
    logger.info("Health check requested. healthy=%s", healthy)
    if healthy:
        return jsonify({"status": "ok"})
    return jsonify({"status": "error", "details": MODEL_LOAD_ERROR}), 503


@app.route("/parse", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}
    user_input = data.get("message", "")
    logger.info("Received /parse request. payload_keys=%s", list(data.keys()))
    return get_response(user_input)


@app.errorhandler(Exception)
def handle_exception(error):
    logger.exception("Unhandled exception while processing request: %s", error)
    return jsonify({"error": "internal_server_error"}), 500


if __name__ == "__main__":
    app.run(debug=True)
